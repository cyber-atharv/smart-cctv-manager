import cv2
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger("cctv.pose")

# COCO / YOLO-Pose Keypoint Index Mapping
YOLO_KEYPOINT_MAP = {
    0: "nose",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
}

class PoseEstimator:
    """
    Modular Human Pose & Keypoint Estimator supporting:
    1. Ultralytics YOLOv8-Pose (yolov8n-pose.pt) for accurate, real-time keypoint extraction
    2. MediaPipe Pose Landmarker
    3. Geometric Approximation Fallback for occluded/edge conditions
    """
    @property
    def backend(self) -> str:
        return self.backend_name
    def __init__(self, min_detection_confidence: float = 0.45, min_tracking_confidence: float = 0.45):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.backend_name = "Geometric Fallback"
        self.yolo_pose = None
        self.mp_pose = None
        self._last_frame_ref = None
        self._cached_results = None

        # 1. Attempt to load YOLOv8-Pose model
        try:
            from ultralytics import YOLO
            self.yolo_pose = YOLO("yolov8n-pose.pt")
            self.backend_name = "YOLOv8-Pose"
            logger.info("YOLOv8-Pose model loaded successfully as primary pose estimator.")
            return
        except Exception as e:
            logger.warning(f"YOLOv8-Pose unavailable ({e}). Checking MediaPipe...")

        # 2. Attempt MediaPipe
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
                self.mp_pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence
                )
                self.backend_name = "MediaPipe Pose"
                logger.info("MediaPipe Pose initialized successfully.")
                return
        except Exception as e:
            logger.warning(f"MediaPipe Pose unavailable ({e}).")

        self.backend_name = "Geometric Fallback"
        logger.info("Operating in Geometric Fallback Pose Estimation mode.")

    def estimate(self, frame: np.ndarray, bbox: Optional[List[int]] = None) -> Optional[Dict[str, Any]]:
        """
        Extracts pose landmarks from frame or ROI bounding box [x1, y1, x2, y2].
        Returns:
          - 'landmarks': dict mapping joint name to {'x': float, 'y': float, 'z': float, 'visibility': float}
          - 'keypoints_found': int
          - 'backend': str
          - 'success': bool
        """
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]

        # 1. YOLOv8-Pose Execution
        if self.yolo_pose is not None:
            try:
                # Run YOLO-Pose inference (cached per frame with optimized 320x320 imgsz for real-time 30+ FPS)
                if self._last_frame_ref is frame and self._cached_results is not None:
                    results = self._cached_results
                else:
                    results = self.yolo_pose(frame, imgsz=320, verbose=False, conf=self.min_detection_confidence)
                    self._last_frame_ref = frame
                    self._cached_results = results
                if results and len(results) > 0 and results[0].keypoints is not None:
                    kpts_data = results[0].keypoints
                    if kpts_data.xy is not None and len(kpts_data.xy) > 0:
                        # If bbox provided, find keypoints closest to bbox center
                        best_idx = 0
                        if bbox is not None and len(kpts_data.xy) > 1:
                            bcx = (bbox[0] + bbox[2]) / 2.0
                            bcy = (bbox[1] + bbox[3]) / 2.0
                            min_dist = float('inf')
                            for k_idx, xy_arr in enumerate(kpts_data.xy):
                                if len(xy_arr) > 0:
                                    k_cx = float(np.mean(xy_arr[:, 0].cpu().numpy()))
                                    k_cy = float(np.mean(xy_arr[:, 1].cpu().numpy()))
                                    dist = (bcx - k_cx)**2 + (bcy - k_cy)**2
                                    if dist < min_dist:
                                        min_dist = dist
                                        best_idx = k_idx

                        xy = kpts_data.xy[best_idx].cpu().numpy()
                        conf = kpts_data.conf[best_idx].cpu().numpy() if kpts_data.conf is not None else None

                        landmarks_dict = {}
                        for coco_idx, name in YOLO_KEYPOINT_MAP.items():
                            if coco_idx < len(xy):
                                px, py = float(xy[coco_idx][0]), float(xy[coco_idx][1])
                                p_conf = float(conf[coco_idx]) if conf is not None and coco_idx < len(conf) else 0.8
                                if px > 0 and py > 0:
                                    landmarks_dict[name] = {
                                        "x": max(0.0, min(1.0, px / float(w))),
                                        "y": max(0.0, min(1.0, py / float(h))),
                                        "z": 0.0,
                                        "visibility": max(0.0, min(1.0, p_conf))
                                    }

                        if len(landmarks_dict) >= 5:
                            # Require at least one shoulder to be visible to reject non-human textures
                            has_shoulder = (
                                landmarks_dict.get("left_shoulder", {}).get("visibility", 0) > 0.35 or
                                landmarks_dict.get("right_shoulder", {}).get("visibility", 0) > 0.35
                            )
                            if has_shoulder:
                                return {
                                    "landmarks": landmarks_dict,
                                    "keypoints_found": len(landmarks_dict),
                                    "backend": self.backend_name,
                                    "success": True
                                }
            except Exception as e:
                logger.debug(f"YOLO-Pose frame inference warning: {e}")

        # 2. MediaPipe Pose Execution (Fallback when YOLO-Pose is offline)
        if self.mp_pose is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_res = self.mp_pose.process(rgb_frame)
                if mp_res and mp_res.pose_landmarks:
                    lms = mp_res.pose_landmarks.landmark
                    # Verify shoulders exist
                    if len(lms) >= 33 and (lms[11].visibility > 0.35 or lms[12].visibility > 0.35):
                        landmarks_dict = {
                            "nose": {"x": lms[0].x, "y": lms[0].y, "z": lms[0].z, "visibility": lms[0].visibility},
                            "left_shoulder": {"x": lms[11].x, "y": lms[11].y, "z": lms[11].z, "visibility": lms[11].visibility},
                            "right_shoulder": {"x": lms[12].x, "y": lms[12].y, "z": lms[12].z, "visibility": lms[12].visibility},
                            "left_elbow": {"x": lms[13].x, "y": lms[13].y, "z": lms[13].z, "visibility": lms[13].visibility},
                            "right_elbow": {"x": lms[14].x, "y": lms[14].y, "z": lms[14].z, "visibility": lms[14].visibility},
                            "left_wrist": {"x": lms[15].x, "y": lms[15].y, "z": lms[15].z, "visibility": lms[15].visibility},
                            "right_wrist": {"x": lms[16].x, "y": lms[16].y, "z": lms[16].z, "visibility": lms[16].visibility},
                            "left_hip": {"x": lms[23].x, "y": lms[23].y, "z": lms[23].z, "visibility": lms[23].visibility},
                            "right_hip": {"x": lms[24].x, "y": lms[24].y, "z": lms[24].z, "visibility": lms[24].visibility},
                            "left_knee": {"x": lms[25].x, "y": lms[25].y, "z": lms[25].z, "visibility": lms[25].visibility},
                            "right_knee": {"x": lms[26].x, "y": lms[26].y, "z": lms[26].z, "visibility": lms[26].visibility},
                            "left_ankle": {"x": lms[27].x, "y": lms[27].y, "z": lms[27].z, "visibility": lms[27].visibility},
                            "right_ankle": {"x": lms[28].x, "y": lms[28].y, "z": lms[28].z, "visibility": lms[28].visibility},
                        }
                        return {
                            "landmarks": landmarks_dict,
                            "keypoints_found": len(landmarks_dict),
                            "backend": "MediaPipe Pose",
                            "success": True
                        }
            except Exception as e:
                logger.debug(f"MediaPipe inference warning: {e}")

        # 3. Strictly return None when no real biological human anatomy is present
        # Prevents ghost / phantom skeleton hallucination on blinds, walls, and inanimate objects
        return None
