import cv2
import time
import numpy as np
import threading
from typing import Optional, Tuple
import logging

logger = logging.getLogger("cctv.camera")

class SyntheticCCTVGenerator:
    """
    Generates a realistic CCTV video stream with animated subjects demonstrating:
    Walking -> Working -> Tripping / Falling -> Lying on Floor -> Recovery.
    Guarantees that demo scenarios (known employee, unknown person, fall incidents)
    can be tested and demonstrated effortlessly on any computer without external video files.
    """
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 15, camera_id: str = "CAM-01"):
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_id = camera_id
        self.frame_count = 0
        self.cycle_length = 300 # ~20 seconds loop at 15 fps

    def get_frame(self) -> np.ndarray:
        self.frame_count += 1
        t = (self.frame_count % self.cycle_length)
        
        # Base CCTV Dark Hallway / Facility Environment
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Gradient floor / ceiling
        for y in range(self.height):
            c_val = int(25 + 20 * (y / self.height))
            frame[y, :] = (c_val, c_val + 5, c_val + 10)

        # Draw CCTV grid perspective lines & hallway borders
        cv2.line(frame, (100, 720), (450, 250), (40, 50, 60), 2)
        cv2.line(frame, (1180, 720), (830, 250), (40, 50, 60), 2)
        cv2.line(frame, (450, 250), (830, 250), (50, 60, 70), 2)
        cv2.rectangle(frame, (450, 100), (830, 250), (35, 45, 55), -1)

        # Animate Person 1 (Primary subject: Employee Rahul / Track 1)
        # Stage 0..80: Walking right to left (Standing)
        # Stage 81..150: Working at workstation
        # Stage 151..210: Sudden slip and Fall down!
        # Stage 211..270: Lying on floor (Confirmed Fall)
        # Stage 271..300: Recovering / Standing up
        
        if t < 80:
            # Walking
            progress = t / 80.0
            px = int(350 + progress * 250)
            py = int(450)
            bw, bh = 80, 200
            activity_state = "walking"
            is_fallen = False
        elif t < 140:
            # Working
            px = 600
            py = int(450 + 5 * np.sin(t * 0.3))
            bw, bh = 85, 190
            activity_state = "working"
            is_fallen = False
        elif t < 170:
            # Falling phase (rapid downward descent and rotation)
            fall_progress = (t - 140) / 30.0
            px = int(600 + fall_progress * 80)
            py = int(450 + fall_progress * 130)
            # Aspect ratio flips from tall (80x200) to wide (190x80)
            bw = int(80 + fall_progress * 110)
            bh = int(200 - fall_progress * 120)
            activity_state = "falling"
            is_fallen = True
        elif t < 260:
            # Lying on floor (Floor persistence)
            px = 680
            py = 580
            bw, bh = 190, 75
            activity_state = "fallen_floor"
            is_fallen = True
        else:
            # Standing back up
            rec_progress = (t - 260) / 40.0
            px = 680
            py = int(580 - rec_progress * 130)
            bw = int(190 - rec_progress * 110)
            bh = int(75 + rec_progress * 125)
            activity_state = "recovering"
            is_fallen = False

        # Draw Person Body / Skeleton representation for detection & pose
        bx1 = px - bw // 2
        by1 = py - bh // 2
        bx2 = px + bw // 2
        by2 = py + bh // 2

        # Draw person figure
        # Torso & Clothing
        shirt_color = (60, 130, 220) if is_fallen else (40, 160, 220)
        pants_color = (120, 80, 50)
        skin_color = (160, 200, 240)

        if not is_fallen:
            # Upright human figure
            # Head
            cv2.circle(frame, (px, by1 + int(bh * 0.15)), int(bw * 0.28), skin_color, -1)
            # Torso
            cv2.rectangle(frame, (bx1 + int(bw * 0.15), by1 + int(bh * 0.25)), (bx2 - int(bw * 0.15), by1 + int(bh * 0.65)), shirt_color, -1)
            # Legs
            cv2.rectangle(frame, (bx1 + int(bw * 0.18), by1 + int(bh * 0.65)), (px - 3, by2), pants_color, -1)
            cv2.rectangle(frame, (px + 3, by1 + int(bh * 0.65)), (bx2 - int(bw * 0.18), by2), pants_color, -1)
            # Arms
            cv2.line(frame, (bx1 + int(bw * 0.15), by1 + int(bh * 0.3)), (bx1 - 5, by1 + int(bh * 0.55)), shirt_color, 8)
            cv2.line(frame, (bx2 - int(bw * 0.15), by1 + int(bh * 0.3)), (bx2 + 5, by1 + int(bh * 0.55)), shirt_color, 8)
        else:
            # Horizontal / Fallen posture on floor
            # Head
            cv2.circle(frame, (bx1 + int(bw * 0.15), py), int(bh * 0.35), skin_color, -1)
            # Torso
            cv2.rectangle(frame, (bx1 + int(bw * 0.25), by1 + int(bh * 0.15)), (bx1 + int(bw * 0.65), by2 - int(bh * 0.15)), shirt_color, -1)
            # Legs extended
            cv2.rectangle(frame, (bx1 + int(bw * 0.65), by1 + int(bh * 0.2)), (bx2, py - 2), pants_color, -1)
            cv2.rectangle(frame, (bx1 + int(bw * 0.65), py + 2), (bx2, by2 - int(bh * 0.2)), pants_color, -1)

        # Add CCTV OSD (On-Screen Display: Camera Name, Timestamp, REC indicator)
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"{self.camera_id} - LIVE", (30, 45), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{timestamp_str}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        # Blinking REC dot
        if (self.frame_count // 10) % 2 == 0:
            cv2.circle(frame, (self.width - 60, 45), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (self.width - 40, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        return frame

class CameraStream:
    """
    Unified Camera Stream Handler supporting Webcam, Video File, RTSP, and Synthetic Feeds.
    Maintains a dedicated reading thread for low-latency non-blocking frame retrieval.
    """
    def __init__(self, camera_id: str, source_type: str = "synthetic", source_uri: str = "0", fps: int = 15):
        self.camera_id = camera_id
        self.source_type = source_type
        self.source_uri = source_uri
        self.target_fps = fps
        self.cap = None
        self.synthetic_gen = None
        self.running = False
        self.current_frame = None
        self.lock = threading.Lock()
        self.thread = None

        if self.source_type == "synthetic":
            self.synthetic_gen = SyntheticCCTVGenerator(camera_id=self.camera_id, fps=self.target_fps)

    def start(self):
        if self.running:
            return
        self.running = True

        if self.source_type in ["webcam", "file", "rtsp"]:
            uri = int(self.source_uri) if (self.source_type == "webcam" and self.source_uri.isdigit()) else self.source_uri
            self.cap = cv2.VideoCapture(uri)
            if not self.cap.isOpened():
                logger.warning(f"Failed to open video source {self.source_uri} for {self.camera_id}. Falling back to synthetic stream.")
                self.source_type = "synthetic"
                self.synthetic_gen = SyntheticCCTVGenerator(camera_id=self.camera_id, fps=self.target_fps)

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started camera capture stream for {self.camera_id} ({self.source_type})")

    def _capture_loop(self):
        interval = 1.0 / float(self.target_fps)
        while self.running:
            start_t = time.time()
            frame = None

            if self.source_type == "synthetic" and self.synthetic_gen is not None:
                frame = self.synthetic_gen.get_frame()
            elif self.cap is not None and self.cap.isOpened():
                ret, captured = self.cap.read()
                if ret and captured is not None:
                    frame = captured
                else:
                    # Video loop for files
                    if self.source_type == "file":
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    else:
                        time.sleep(0.1)

            if frame is not None:
                with self.lock:
                    self.current_frame = frame

            elapsed = time.time() - start_t
            sleep_t = max(0.001, interval - elapsed)
            time.sleep(sleep_t)

    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        logger.info(f"Stopped camera capture stream for {self.camera_id}")
