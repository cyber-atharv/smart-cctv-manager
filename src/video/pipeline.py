import cv2
import time
import threading
import numpy as np
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
import logging

from src.config import config
from src.database.db import SessionLocal
from src.video.camera import CameraStream, mask_camera_uri
from src.events.clip_manager import RollingVideoBuffer
from src.events.event_manager import event_manager
from src.detection.person_detector import PersonDetector
from src.tracking.tracker import MultiObjectTracker
from src.pose.pose_estimator import PoseEstimator
from src.features.feature_extractor import FeatureExtractor
from src.activity.classifier import ActivityClassifier
from src.activity.temporal_verifier import TemporalFallVerifier, FallVerificationState
from src.face.recognizer import FaceRecognizer
from src.safety.ppe_detector import PPEDetector
from src.zones.zone_manager import ZoneManager, Zone

logger = logging.getLogger("cctv.pipeline")

class VisionPipelineWorker:
    """
    End-to-End Modular Computer Vision & Safety Pipeline Worker per Camera Stream.
    Decoupled Architecture:
    1. Camera Capture Stream (Webcam, DroidCam Mobile, RTSP, Video File, Demo)
    2. Optional AI Processing Layer (YOLO -> Tracker -> Pose -> Features -> Activity ML -> Fall Verifier -> Face ID)
    3. Multi-Mode Streaming (Annotated AI Stream / Raw Live Stream)
    """
    def __init__(self, camera_id: str, source_type: str = "webcam", source_uri: str = "0", fps: int = 24, ai_enabled: bool = True):
        self.camera_id = camera_id
        self.source_type = source_type.lower()
        self.source_uri = source_uri
        self.fps = fps
        self.ai_enabled = ai_enabled
        self.stream = CameraStream(camera_id, source_type, source_uri, fps)
        self.video_buffer = RollingVideoBuffer(fps=fps, max_seconds=20)
        
        # Modular computer vision components
        self.detector = PersonDetector(conf_threshold=0.35)
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
        self.ppe_detector = PPEDetector()
        self.zone_manager = ZoneManager(camera_id=camera_id)
        self.draw_zones_enabled = True

        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        self.latest_raw_frame = None
        self.latest_annotated_frame = None
        self.current_tracks_info = []
        self.stats = {
            "fps": 0.0,
            "total_frames": 0,
            "active_tracks": 0,
            "fall_alerts": 0,
            "zone_alerts": 0,
            "ppe_compliant_count": 0,
            "ppe_violation_count": 0
        }
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
        logger.info(f"Vision Pipeline Worker started for {self.camera_id} (AI: {'ACTIVE' if self.ai_enabled else 'STOPPED'})")

    def start_ai(self):
        self.ai_enabled = True
        logger.info(f"AI Analysis enabled for camera {self.camera_id}")

    def stop_ai(self):
        self.ai_enabled = False
        with self.lock:
            self.current_tracks_info = []
        logger.info(f"AI Analysis stopped for camera {self.camera_id}")

    def toggle_ai(self) -> bool:
        self.ai_enabled = not self.ai_enabled
        if not self.ai_enabled:
            with self.lock:
                self.current_tracks_info = []
        return self.ai_enabled

    def reconnect(self):
        self.stream.reconnect()

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
            with self.lock:
                self.latest_raw_frame = raw_frame.copy()

            if self.stream.is_online:
                self.video_buffer.add_frame(raw_frame)

            # If camera is offline or AI analysis is disabled, skip heavy inference
            if not self.stream.is_online:
                with self.lock:
                    self.latest_annotated_frame = raw_frame
                    self.current_tracks_info = []
                    self.stats["active_tracks"] = 0
                time.sleep(0.05)
                continue

            if not self.ai_enabled:
                annotated = raw_frame.copy()
                self._draw_ai_stopped_hud(annotated)
                with self.lock:
                    self.latest_annotated_frame = annotated
                    self.current_tracks_info = []
                    self.stats["active_tracks"] = 0
                time.sleep(0.02)
                continue

            # Periodically sync face gallery every 15 seconds
            if time.time() - self.last_face_sync > 15.0:
                db_sync = SessionLocal()
                try:
                    self.face_recognizer.sync_gallery(db_sync)
                    self.last_face_sync = time.time()
                finally:
                    db_sync.close()

            # 1. Person Detection (YOLO on real frame)
            detections = self.detector.detect(raw_frame)

            # 2. Multi-Object Tracking
            active_tracks = self.tracker.update(detections)

            annotated = raw_frame.copy()
            h, w = annotated.shape[:2]
            tracks_info = []

            # 3. Process Each Tracked Individual
            for track in active_tracks:
                # 3.1 Pose Estimation
                pose_data = self.pose_estimator.estimate(raw_frame, bbox=track.bbox)
                landmarks = pose_data["landmarks"] if pose_data else {}

                # 3.2 Feature Extraction
                features = self.feature_extractor.extract(
                    landmarks=landmarks,
                    bbox=track.bbox,
                    frame_shape=raw_frame.shape,
                    velocity=track.velocity
                )

                # 3.3 Activity Classification (Random Forest / Decision Tree)
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

                # 3.5 Face Recognition (Decoupled: Unknown persons still protected)
                if track.employee_id is None or track.age_frames % 12 == 0:
                    emp_id, emp_code, emp_name, face_conf = self.face_recognizer.recognize_in_person_roi(
                        raw_frame, person_bbox=track.bbox
                    )
                    if emp_id is not None:
                        track.set_identity(emp_id, emp_code, emp_name, face_conf)

                # 3.6 PPE (Personal Protective Equipment) Inspection
                ppe_data = self.ppe_detector.detect_ppe(raw_frame, bbox=track.bbox, landmarks=landmarks)
                track.set_ppe_status(ppe_data)

                # 3.7 Event Trigger on Confirmed Fall Incident
                if should_trigger_event:
                    self.stats["fall_alerts"] += 1
                    logger.warning(f"DISPATCHING REAL FALL EVENT for Track {track.track_id} on {self.camera_id}")
                    
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

            # 4. Multi-Domain Zone & ROI Processing
            self.zone_manager.update(active_tracks, raw_frame.shape)

            # Map zone membership back to tracks
            track_map = {t.track_id: t for t in active_tracks}
            for z in self.zone_manager.zones.values():
                for tid in z.occupant_track_ids:
                    if tid in track_map:
                        track_map[tid].current_zone_id = z.id
                        track_map[tid].current_zone_name = z.name
                        track_map[tid].current_zone_type = z.zone_type

                # Trigger Security Event on Zone Alert (Restricted Intrusion / Loitering)
                if z.is_alert:
                    last_alert = getattr(z, "_last_db_alert_time", 0.0)
                    now_ts = time.time()
                    if now_ts - last_alert > 20.0:
                        z._last_db_alert_time = now_ts
                        self.stats["zone_alerts"] += 1
                        db_z_event = SessionLocal()
                        try:
                            event_manager.create_event(
                                db=db_z_event,
                                camera_id=self.camera_id,
                                track_id=z.occupant_track_ids[0] if z.occupant_track_ids else None,
                                employee_id=None,
                                activity=f"zone_{z.zone_type}",
                                activity_confidence=0.94,
                                identity_confidence=0.0,
                                identity_status="unknown",
                                current_frame=raw_frame,
                                video_buffer=self.video_buffer,
                                notes=f"Zone Security Alert: {z.name} - {z.alert_message}"
                            )
                        except Exception as err:
                            logger.error(f"Error persisting zone security event: {err}")
                        finally:
                            db_z_event.close()

            # 5. Render Visual HUD Layers
            # 5.1 Draw Zones on Frame
            if self.draw_zones_enabled:
                self.zone_manager.draw_zones(annotated)

            # 5.2 Draw Person Overlays on Frame
            for track in active_tracks:
                self._draw_person_overlay(annotated, track, {}, {})
                tracks_info.append(track.to_dict())

            # 5.3 Draw Top Global HUD
            self._draw_stats_hud(annotated, len(active_tracks))

            # Store updated state
            with self.lock:
                self.latest_annotated_frame = annotated
                self.current_tracks_info = tracks_info
                self.stats["total_frames"] = frame_counter
                self.stats["active_tracks"] = len(active_tracks)
                self.stats["ppe_compliant_count"] = sum(1 for t in active_tracks if t.ppe_compliant)
                self.stats["ppe_violation_count"] = sum(1 for t in active_tracks if not t.ppe_compliant)

            # Compute effective FPS
            now = time.time()
            dt = now - last_time
            if dt >= 1.0:
                self.stats["fps"] = round(frame_counter / dt, 1)
                frame_counter = 0
                last_time = now

            elapsed = time.time() - start_loop
            sleep_time = max(0.002, frame_interval - elapsed)
            time.sleep(sleep_time)

    def _draw_person_overlay(self, frame: np.ndarray, track: Any, landmarks: Dict, features: Dict):
        """Draws bounding box, skeleton, identity tag, activity, PPE status, and zone badge."""
        x1, y1, x2, y2 = track.bbox
        h, w = frame.shape[:2]

        if track.fall_stage == FallVerificationState.CONFIRMED_FALL:
            box_color = (0, 0, 255) # Red
            badge_text = "ALERT: FALL CONFIRMED"
        elif track.fall_stage in [FallVerificationState.POSSIBLE_FALL, FallVerificationState.SUSPICIOUS]:
            box_color = (0, 165, 255) # Orange
            badge_text = f"WARNING: {track.fall_stage}"
        elif not track.ppe_compliant and track.current_zone_type == "restricted":
            box_color = (0, 80, 255) # Warning Red-Orange
            badge_text = "PPE VIOLATION: NO HELMET/VEST"
        else:
            box_color = (0, 255, 128) if track.identity_status == "recognized" else (220, 200, 50)
            badge_text = f"PERSON #{track.track_id}"

        # Bounding box with corner accents
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        corner_len = min(20, max(6, (x2 - x1) // 4))
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), box_color, 4)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), box_color, 4)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_color, 4)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), box_color, 4)

        # Labels: Identity + Activity
        identity_label = f"ID: {track.employee_code or 'Unknown'} ({track.employee_name or 'Unregistered'})"
        activity_label = f"Act: {track.current_activity.upper()} ({int(track.activity_confidence * 100)}%)"

        # PPE Compliance Tag
        if track.ppe_compliant:
            ppe_label = "PPE: OK [Helmet+Vest]"
            ppe_color = (0, 230, 118)
        else:
            v_short = track.ppe_violations[0] if track.ppe_violations else "Missing PPE"
            ppe_label = f"PPE: ⚠ {v_short}"
            ppe_color = (0, 140, 255)

        # Zone Tag
        zone_label = f"Zone: {track.current_zone_name}" if track.current_zone_name else ""

        label_y = max(38, y1 - 10)
        box_height = 42 if not zone_label else 54
        cv2.rectangle(frame, (x1, label_y - box_height), (x1 + 250, label_y + 4), (15, 18, 25), -1)
        cv2.rectangle(frame, (x1, label_y - box_height), (x1 + 250, label_y + 4), box_color, 1)
        
        cv2.putText(frame, identity_label, (x1 + 6, label_y - box_height + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, activity_label, (x1 + 6, label_y - box_height + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.40, box_color, 1, cv2.LINE_AA)
        cv2.putText(frame, ppe_label, (x1 + 6, label_y - box_height + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.38, ppe_color, 1, cv2.LINE_AA)
        if zone_label:
            cv2.putText(frame, zone_label, (x1 + 6, label_y - box_height + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)

        if track.fall_stage != FallVerificationState.NORMAL:
            cv2.rectangle(frame, (x1, y1 - box_height - 24), (x1 + 250, y1 - box_height - 4), box_color, -1)
            cv2.putText(frame, badge_text, (x1 + 8, y1 - box_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)

    def _draw_stats_hud(self, frame: np.ndarray, num_tracks: int):
        """Draws top HUD banner on annotated frames with multi-domain telemetry."""
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 32), (12, 16, 24), -1)
        cv2.line(frame, (0, 32), (w, 32), (40, 50, 65), 1)

        is_demo = (self.source_type == "demo")
        stream_tag = "[DEMO SIMULATION]" if is_demo else "[LIVE CAMERA]"
        tag_color = (180, 100, 255) if is_demo else (0, 255, 0)

        cv2.putText(frame, f"{self.camera_id} {stream_tag}", (15, 22), cv2.FONT_HERSHEY_DUPLEX, 0.52, tag_color, 1, cv2.LINE_AA)

        fps_val = self.stats.get('fps', self.fps)
        zones_summary = self.zone_manager.get_summary()
        occ_tables = zones_summary["hospitality"]["occupied_tables"]
        tot_tables = zones_summary["hospitality"]["total_tables"]
        queue_count = zones_summary["queue_analytics"]["total_waiting_people"]
        
        hud_text = f"FPS: {fps_val} | People: {num_tracks} | Tables: {occ_tables}/{tot_tables} | Queue: {queue_count} | Falls: {self.stats.get('fall_alerts', 0)}"
        cv2.putText(frame, hud_text, (max(200, w - 540), 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 229, 255), 1, cv2.LINE_AA)

    def _draw_ai_stopped_hud(self, frame: np.ndarray):
        """Draws clean HUD banner when AI processing is paused."""
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 32), (12, 16, 24), -1)
        cv2.line(frame, (0, 32), (w, 32), (40, 50, 65), 1)

        is_demo = (self.source_type == "demo")
        stream_tag = "[DEMO SIMULATION]" if is_demo else "[LIVE CAMERA]"
        tag_color = (180, 100, 255) if is_demo else (0, 255, 0)

        cv2.putText(frame, f"{self.camera_id} {stream_tag}", (15, 22), cv2.FONT_HERSHEY_DUPLEX, 0.55, tag_color, 1, cv2.LINE_AA)
        cv2.putText(frame, "AI ANALYSIS: STOPPED (Raw Video Feed)", (max(220, w - 380), 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 160, 175), 1, cv2.LINE_AA)

    def get_latest_frame(self, mode: str = "ai") -> Optional[np.ndarray]:
        with self.lock:
            if mode == "raw" and self.latest_raw_frame is not None:
                return self.latest_raw_frame.copy()
            if self.latest_annotated_frame is not None:
                return self.latest_annotated_frame.copy()
            if self.latest_raw_frame is not None:
                return self.latest_raw_frame.copy()
            return None

    def get_mjpeg_generator(self, mode: str = "ai"):
        """Yields MJPEG multipart frame stream for browser HTML5 <img> embedding."""
        interval = 1.0 / max(5, float(self.fps))
        while self.running:
            frame = self.get_latest_frame(mode=mode)
            if frame is not None:
                ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(interval)

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.stream.stop()
        logger.info(f"Vision Pipeline Worker stopped for {self.camera_id}")
