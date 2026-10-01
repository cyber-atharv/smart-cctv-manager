import cv2
import time
import threading
import numpy as np
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
import logging

from src.config import config
from src.database.db import SessionLocal
from src.video.camera import CameraStream
from src.events.clip_manager import RollingVideoBuffer
from src.events.event_manager import event_manager
from src.detection.person_detector import PersonDetector
from src.tracking.tracker import MultiObjectTracker
from src.pose.pose_estimator import PoseEstimator
from src.features.feature_extractor import FeatureExtractor
from src.activity.classifier import ActivityClassifier
from src.activity.temporal_verifier import TemporalFallVerifier, FallVerificationState
from src.face.recognizer import FaceRecognizer

logger = logging.getLogger("cctv.pipeline")

class VisionPipelineWorker:
    """
    End-to-End Modular Computer Vision & Safety Pipeline Worker per Camera Stream.
    Processes: Video Stream -> Person Detection -> Tracking -> Pose -> Features -> Activity ML -> Fall Verifier -> Face ID -> Event Service.
    """
    def __init__(self, camera_id: str, source_type: str = "synthetic", source_uri: str = "0", fps: int = 15):
        self.camera_id = camera_id
        self.fps = fps
        self.stream = CameraStream(camera_id, source_type, source_uri, fps)
        self.video_buffer = RollingVideoBuffer(fps=fps, max_seconds=20)
        
        # Modular vision pipeline components
        self.detector = PersonDetector(conf_threshold=0.40)
        self.tracker = MultiObjectTracker(camera_id=camera_id, iou_threshold=0.3, max_age=30)
        self.pose_estimator = PoseEstimator(min_detection_confidence=0.4, min_tracking_confidence=0.4)
        self.feature_extractor = FeatureExtractor()
        self.activity_classifier = ActivityClassifier()
        self.fall_verifier = TemporalFallVerifier(
            suspicious_frames_threshold=3,
            possible_fall_frames_threshold=6,
            floor_persistence_threshold=12,
            fall_confidence_threshold=0.70,
            debounce_seconds=8.0
        )
        self.face_recognizer = FaceRecognizer(match_threshold=0.55)

        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        self.latest_annotated_frame = None
        self.current_tracks_info = []
        self.stats = {"fps": 0.0, "total_frames": 0, "active_tracks": 0, "fall_alerts": 0}
        self.last_face_sync = 0.0

    def start(self):
        if self.running:
            return
        self.running = True
        self.stream.start()
        
        # Initial face gallery sync
        db = SessionLocal()
        try:
            self.face_recognizer.sync_gallery(db)
        finally:
            db.close()

        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        logger.info(f"Vision Pipeline Worker started for {self.camera_id}")

    def _worker_loop(self):
        frame_interval = 1.0 / float(self.fps)
        last_time = time.time()
        frame_counter = 0

        while self.running:
            start_loop = time.time()
            raw_frame = self.stream.read()
            if raw_frame is None:
                time.sleep(0.02)
                continue

            frame_counter += 1
            self.video_buffer.add_frame(raw_frame)

            # Periodically sync face gallery every 15 seconds
            if time.time() - self.last_face_sync > 15.0:
                db_sync = SessionLocal()
                try:
                    self.face_recognizer.sync_gallery(db_sync)
                    self.last_face_sync = time.time()
                finally:
                    db_sync.close()

            # 1. Person Detection
            detections = self.detector.detect(raw_frame)

            # 2. Multi-Object Tracking
            active_tracks = self.tracker.update(detections)

            annotated = raw_frame.copy()
            h, w = annotated.shape[:2]
            tracks_info = []

            # 3. Process Each Tracked Individual
            for track in active_tracks:
                # 3.1 Pose Estimation (on person ROI)
                pose_data = self.pose_estimator.estimate(raw_frame, bbox=track.bbox)
                landmarks = pose_data["landmarks"] if pose_data else {}

                # 3.2 Feature Extraction
                features = self.feature_extractor.extract(
                    landmarks=landmarks,
                    bbox=track.bbox,
                    frame_shape=raw_frame.shape,
                    velocity=track.velocity
                )

                # 3.3 Activity Classification
                predicted_act, act_conf, probs = self.activity_classifier.predict(features)
                track.current_activity = predicted_act
                track.activity_confidence = act_conf

                # 3.4 Temporal Fall Verification State Machine
                fall_stage, should_trigger_event, aggregated_conf = self.fall_verifier.process_frame(
                    track_id=track.track_id,
                    predicted_activity=predicted_act,
                    activity_confidence=act_conf,
                    features=features
                )
                track.fall_stage = fall_stage

                # 3.5 Face Recognition (Optional Identity Layer)
                if track.employee_id is None or track.age_frames % 10 == 0:
                    emp_id, emp_code, emp_name, face_conf = self.face_recognizer.recognize_in_person_roi(
                        raw_frame, person_bbox=track.bbox
                    )
                    if emp_id is not None:
                        track.set_identity(emp_id, emp_code, emp_name, face_conf)

                # 3.6 Event Trigger on Confirmed Fall
                if should_trigger_event:
                    self.stats["fall_alerts"] += 1
                    logger.warning(f"DISPATCHING FALL EVENT for Track {track.track_id} on {self.camera_id}")
                    
                    db_event = SessionLocal()
                    try:
                        event_manager.create_event(
                            db=db_event,
                            camera_id=self.camera_id,
                            track_id=track.track_id,
                            employee_id=track.employee_id,
                            activity="falling",
                            activity_confidence=act_conf,
                            identity_confidence=track.identity_confidence,
                            identity_status=track.identity_status,
                            current_frame=raw_frame,
                            video_buffer=self.video_buffer,
                            notes=f"Temporal Fall Confirmed (Stage: {fall_stage}, Body Angle: {features.get('body_angle')} deg)"
                        )
                    except Exception as err:
                        logger.error(f"Error persisting fall event: {err}")
                    finally:
                        db_event.close()

                # 3.7 Annotate Visual HUD on Frame
                self._draw_person_overlay(annotated, track, landmarks, features)
                tracks_info.append(track.to_dict())

            # Draw Global Pipeline Stats HUD
            self._draw_stats_hud(annotated, len(active_tracks))

            # Store updated state
            with self.lock:
                self.latest_annotated_frame = annotated
                self.current_tracks_info = tracks_info
                self.stats["total_frames"] = frame_counter
                self.stats["active_tracks"] = len(active_tracks)

            # Compute effective FPS
            now = time.time()
            dt = now - last_time
            if dt >= 1.0:
                self.stats["fps"] = round(frame_counter / dt, 1)
                frame_counter = 0
                last_time = now

            elapsed = time.time() - start_loop
            sleep_time = max(0.001, frame_interval - elapsed)
            time.sleep(sleep_time)

    def _draw_person_overlay(self, frame: np.ndarray, track: Any, landmarks: Dict, features: Dict):
        """Draws bounding box, skeleton, identity tag, and safety alert status badges."""
        x1, y1, x2, y2 = track.bbox
        h, w = frame.shape[:2]

        # Determine theme color based on safety state
        if track.fall_stage == FallVerificationState.CONFIRMED_FALL:
            box_color = (0, 0, 255) # Red
            badge_text = "ALERT: FALL CONFIRMED"
        elif track.fall_stage in [FallVerificationState.POSSIBLE_FALL, FallVerificationState.SUSPICIOUS]:
            box_color = (0, 165, 255) # Orange
            badge_text = f"WARNING: {track.fall_stage}"
        else:
            box_color = (0, 255, 128) if track.identity_status == "recognized" else (220, 200, 50)
            badge_text = f"TRACK #{track.track_id}"

        # Bounding box with corner accents
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        corner_len = min(20, (x2 - x1) // 4)
        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), box_color, 4)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), box_color, 4)
        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_color, 4)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), box_color, 4)

        # Draw Pose Landmarks and Skeleton if available
        if landmarks:
            for name, lm in landmarks.items():
                lx, ly = int(lm["x"] * w), int(lm["y"] * h)
                cv2.circle(frame, (lx, ly), 3, (0, 255, 255), -1)

        # Label Badges: Identity + Activity
        identity_label = f"ID: {track.employee_code or 'Unknown'} ({track.employee_name or 'Unregistered'})"
        activity_label = f"Act: {track.current_activity.upper()} ({int(track.activity_confidence * 100)}%)"

        label_y = max(25, y1 - 10)
        # Background box for readable text
        cv2.rectangle(frame, (x1, label_y - 22), (x1 + 240, label_y + 18), (20, 20, 20), -1)
        cv2.rectangle(frame, (x1, label_y - 22), (x1 + 240, label_y + 18), box_color, 1)
        
        cv2.putText(frame, identity_label, (x1 + 5, label_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, activity_label, (x1 + 5, label_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1, cv2.LINE_AA)

        # Fall Stage Alert Banner above bbox if non-normal
        if track.fall_stage != FallVerificationState.NORMAL:
            cv2.rectangle(frame, (x1, y1 - 45), (x1 + 240, y1 - 25), box_color, -1)
            cv2.putText(frame, badge_text, (x1 + 8, y1 - 31), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)

    def _draw_stats_hud(self, frame: np.ndarray, num_tracks: int):
        """Draws top HUD status banner with AI pipeline metadata."""
        h, w = frame.shape[:2]
        # Top banner overlay
        cv2.rectangle(frame, (0, 0), (w, 30), (15, 15, 20), -1)
        cv2.line(frame, (0, 30), (w, 30), (40, 50, 60), 1)

        hud_text = (
            f"AI Pipeline: ACTIVE | Cam: {self.camera_id} | FPS: {self.stats.get('fps', 15.0)} | "
            f"People: {num_tracks} | Fall Alerts: {self.stats.get('fall_alerts', 0)}"
        )
        cv2.putText(frame, hud_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.latest_annotated_frame.copy() if self.latest_annotated_frame is not None else None

    def get_mjpeg_generator(self):
        """Yields MJPEG multipart frame stream for direct browser HTML5 <img> embedding."""
        while self.running:
            frame = self.get_latest_frame()
            if frame is not None:
                ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(1.0 / float(self.fps))

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.stream.stop()
        logger.info(f"Vision Pipeline Worker stopped for {self.camera_id}")
