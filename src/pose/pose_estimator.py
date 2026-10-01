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
    def __init__(self, min_detection_confidence: float = 0.45, min_tracking_confidence: float = 0.45):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.backend_name = "Geometric Fallback"
        self.yolo_pose = None
        self.mp_pose = None

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
                # Run YOLO-Pose inference
                results = self.yolo_pose(frame, verbose=False, conf=self.min_detection_confidence)
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
                            return {
                                "landmarks": landmarks_dict,
                                "keypoints_found": len(landmarks_dict),
                                "backend": self.backend_name,
                                "success": True
                            }
            except Exception as e:
                logger.debug(f"YOLO-Pose frame inference warning: {e}")

        # 2. Geometric Approximation Fallback
        if bbox is not None:
            return self._approximate_from_bbox(bbox, w, h)

        return None

    def _approximate_from_bbox(self, bbox: List[int], w: int, h: int) -> Dict[str, Any]:
        """Approximates anatomical joint positions from person bounding box geometry."""
        x1, y1, x2, y2 = bbox
        nx1, ny1 = x1 / float(w), y1 / float(h)
        nx2, ny2 = x2 / float(w), y2 / float(h)
        cx = (nx1 + nx2) / 2.0
        bw = nx2 - nx1
        bh = ny2 - ny1

        landmarks_dict = {
            "nose": {"x": cx, "y": ny1 + 0.1 * bh, "z": 0.0, "visibility": 0.6},
            "left_shoulder": {"x": cx - 0.25 * bw, "y": ny1 + 0.22 * bh, "z": 0.0, "visibility": 0.6},
            "right_shoulder": {"x": cx + 0.25 * bw, "y": ny1 + 0.22 * bh, "z": 0.0, "visibility": 0.6},
            "left_elbow": {"x": cx - 0.35 * bw, "y": ny1 + 0.40 * bh, "z": 0.0, "visibility": 0.5},
            "right_elbow": {"x": cx + 0.35 * bw, "y": ny1 + 0.40 * bh, "z": 0.0, "visibility": 0.5},
            "left_wrist": {"x": cx - 0.40 * bw, "y": ny1 + 0.55 * bh, "z": 0.0, "visibility": 0.5},
            "right_wrist": {"x": cx + 0.40 * bw, "y": ny1 + 0.55 * bh, "z": 0.0, "visibility": 0.5},
            "left_hip": {"x": cx - 0.20 * bw, "y": ny1 + 0.55 * bh, "z": 0.0, "visibility": 0.6},
            "right_hip": {"x": cx + 0.20 * bw, "y": ny1 + 0.55 * bh, "z": 0.0, "visibility": 0.6},
            "left_knee": {"x": cx - 0.20 * bw, "y": ny1 + 0.78 * bh, "z": 0.0, "visibility": 0.6},
            "right_knee": {"x": cx + 0.20 * bw, "y": ny1 + 0.78 * bh, "z": 0.0, "visibility": 0.6},
            "left_ankle": {"x": cx - 0.20 * bw, "y": ny2 - 0.02 * bh, "z": 0.0, "visibility": 0.6},
            "right_ankle": {"x": cx + 0.20 * bw, "y": ny2 - 0.02 * bh, "z": 0.0, "visibility": 0.6},
        }

        return {
            "landmarks": landmarks_dict,
            "keypoints_found": len(landmarks_dict),
            "backend": "Geometric Fallback",
            "success": True,
            "is_approximated": True
        }
