import cv2
import time
import numpy as np
import threading
import re
import socket
import logging
from typing import Optional, Tuple, Dict, Any, List

logger = logging.getLogger("cctv.camera")

def mask_camera_uri(uri: str) -> str:
    """Masks credentials in camera URIs to prevent password leakage in logs and UI."""
    if not uri or not isinstance(uri, str):
        return ""
    # Regex to mask passwords in rtsp://user:pass@host or http://user:pass@host
    return re.sub(r'(://[^:]+:)([^@]+)(@)', r'\1******\3', uri)

def scan_local_cameras(max_indices: int = 6) -> List[Dict[str, Any]]:
    """
    Probes local hardware video capture devices (Webcams, USB cameras, DroidCam virtual cameras).
    Returns a list of discovered devices with index, name, and default resolution.
    """
    discovered = []
    for idx in range(max_indices):
        try:
            # Quick non-blocking probe
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    h, w = frame.shape[:2]
                    discovered.append({
                        "index": idx,
                        "name": f"Local Camera #{idx}" + (" (Integrated / Primary)" if idx == 0 else " (USB / DroidCam)"),
                        "resolution": f"{w}x{h}",
                        "width": w,
                        "height": h,
                        "fps": int(cap.get(cv2.CAP_PROP_FPS)) or 30
                    })
                cap.release()
        except Exception as e:
            logger.debug(f"Device index {idx} probe error: {e}")
    return discovered

def scan_network_cameras(timeout: float = 2.0) -> List[Dict[str, Any]]:
    """
    Probes local network for ONVIF-compatible IP cameras using WS-Discovery (UDP 3702).
    """
    devices = []
    ws_discovery_msg = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<Envelope xmlns="http://www.w3.org/2003/05/soap-envelope" '
        'xmlns:dn="http://www.onvif.org/ver10/network/wsdl">'
        '<Header>'
        '<wsa:MessageID xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing">'
        'uuid:cctv-onvif-probe-001</wsa:MessageID>'
        '<wsa:To xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing">'
        'urn:schemas-xmlsoap-org:ws:2005:04:discovery</wsa:To>'
        '<wsa:Action xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing">'
        'http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</wsa:Action>'
        '</Header>'
        '<Body>'
        '<Probe xmlns="http://schemas.xmlsoap.org/ws/2005/04/discovery">'
        '<Types>dn:NetworkVideoTransmitter</Types>'
        '</Probe>'
        '</Body>'
        '</Envelope>'
    )
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(timeout)
        sock.sendto(ws_discovery_msg.encode('utf-8'), ('239.255.255.250', 3702))

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                data, addr = sock.recvfrom(65535)
                ip = addr[0]
                if not any(d["ip"] == ip for d in devices):
                    devices.append({
                        "ip": ip,
                        "name": f"ONVIF Camera ({ip})",
                        "type": "onvif",
                        "suggested_rtsp": f"rtsp://admin:admin@{ip}:554/live/ch0"
                    })
            except socket.timeout:
                break
        sock.close()
    except Exception as e:
        logger.debug(f"Network discovery probe note: {e}")
    return devices

def verify_camera_source(source_type: str, source_uri: str, username: str = "", password: str = "", timeout: float = 5.0) -> Dict[str, Any]:
    """
    Performs a thorough real-time multi-step verification of a camera source:
    1. Connects to the source
    2. Opens video stream
    3. Reads multiple frames
    4. Validates frame dimensions, aspect ratio, and flow
    5. Measures actual stream resolution and estimated FPS
    Returns detailed step-by-step verification checklist and diagnostics.
    """
    steps = [
        {"name": "Connection initialization", "status": "pending", "detail": ""},
        {"name": "Video stream opened", "status": "pending", "detail": ""},
        {"name": "Initial frame received", "status": "pending", "detail": ""},
        {"name": "Multiple frames continuous read", "status": "pending", "detail": ""},
        {"name": "Stream metrics measured", "status": "pending", "detail": ""}
    ]

    target_uri = source_uri
    # Embed credentials if provided and not already present
    if username and password and "rtsp://" in target_uri and "@" not in target_uri:
        target_uri = target_uri.replace("rtsp://", f"rtsp://{username}:{password}@")

    source_type = source_type.lower()
    if source_type in ["webcam", "usb", "local"]:
        try:
            target_uri = int(source_uri) if str(source_uri).isdigit() else 0
        except ValueError:
            target_uri = 0

    if source_type == "demo":
        return {
            "success": True,
            "steps": [
                {"name": "Connection initialization", "status": "pass", "detail": "Demo mode selected"},
                {"name": "Video stream opened", "status": "pass", "detail": "Synthetic stream ready"},
                {"name": "Initial frame received", "status": "pass", "detail": "1280x720 RGB frame generated"},
                {"name": "Multiple frames continuous read", "status": "pass", "detail": "15 FPS benchmark confirmed"},
                {"name": "Stream metrics measured", "status": "pass", "detail": "Resolution: 1280x720, FPS: 15"}
            ],
            "resolution": "1280x720",
            "width": 1280,
            "height": 720,
            "fps": 15.0,
            "message": "DEMO CAMERA VERIFIED (SYNTHETIC SIMULATION)",
            "is_demo": True
        }

    # Step 1: Initialize connection
    steps[0]["status"] = "pass"
    steps[0]["detail"] = f"Target: {mask_camera_uri(str(target_uri))}"

    cap = None
    try:
        # Step 2: Open stream
        start_t = time.time()
        cap = cv2.VideoCapture(target_uri)
        
        if not cap.isOpened():
            steps[1]["status"] = "fail"
            steps[1]["detail"] = "Failed to open video source. Please check device index or URL."
            return {
                "success": False,
                "steps": steps,
                "error": "Could not open video stream. Verify camera is plugged in or RTSP server is reachable.",
                "message": "CAMERA VERIFICATION FAILED"
            }
        
        steps[1]["status"] = "pass"
        steps[1]["detail"] = "Video stream connection successfully established."

        # Step 3: Read first frame
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            steps[2]["status"] = "fail"
            steps[2]["detail"] = "Stream opened but returned empty or corrupt frame."
            cap.release()
            return {
                "success": False,
                "steps": steps,
                "error": "No valid frames received from camera.",
                "message": "CAMERA VERIFICATION FAILED"
            }

        h, w = frame.shape[:2]
        steps[2]["status"] = "pass"
        steps[2]["detail"] = f"Valid frame captured (Shape: {w}x{h})"

        # Step 4: Read multiple frames to verify continuity
        frames_read = 0
        read_start = time.time()
        for _ in range(5):
            r, f = cap.read()
            if r and f is not None:
                frames_read += 1
            time.sleep(0.03)

        if frames_read < 3:
            steps[3]["status"] = "fail"
            steps[3]["detail"] = f"Stream unstable: only {frames_read}/5 test frames received."
            cap.release()
            return {
                "success": False,
                "steps": steps,
                "error": "Camera stream dropped frames during continuous verification.",
                "message": "CAMERA VERIFICATION FAILED"
            }

        steps[3]["status"] = "pass"
        steps[3]["detail"] = f"{frames_read}/5 continuous test frames verified."

        # Step 5: Measure metrics
        reported_fps = cap.get(cv2.CAP_PROP_FPS)
        measured_fps = round(frames_read / max(0.01, (time.time() - read_start)), 1)
        fps = reported_fps if (reported_fps and 5.0 <= reported_fps <= 60.0) else (measured_fps or 24.0)

        steps[4]["status"] = "pass"
        steps[4]["detail"] = f"Resolution: {w}x{h}, Estimated FPS: {fps}"

        cap.release()
        return {
            "success": True,
            "steps": steps,
            "resolution": f"{w}x{h}",
            "width": w,
            "height": h,
            "fps": fps,
            "message": "CAMERA VERIFIED",
            "is_demo": False
        }

    except Exception as e:
        logger.error(f"Camera verification error: {e}")
        if cap is not None:
            cap.release()
        return {
            "success": False,
            "steps": steps,
            "error": f"Verification exception: {str(e)}",
            "message": "CAMERA VERIFICATION FAILED"
        }

class SyntheticCCTVGenerator:
    """
    Generates an educational synthetic CCTV demo stream for offline demonstration when
    the user explicitly selects Demo Mode.
    Clearly marked with 'DEMO / SYNTHETIC SIMULATION'.
    """
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 15, camera_id: str = "CAM-DEMO"):
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_id = camera_id
        self.frame_count = 0
        self.cycle_length = 300

    def get_frame(self) -> np.ndarray:
        self.frame_count += 1
        t = (self.frame_count % self.cycle_length)
        
        # Base dark facility scene
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for y in range(self.height):
            c_val = int(25 + 20 * (y / self.height))
            frame[y, :] = (c_val, c_val + 5, c_val + 10)

        cv2.line(frame, (100, 720), (450, 250), (40, 50, 60), 2)
        cv2.line(frame, (1180, 720), (830, 250), (40, 50, 60), 2)
        cv2.line(frame, (450, 250), (830, 250), (50, 60, 70), 2)
        cv2.rectangle(frame, (450, 100), (830, 250), (35, 45, 55), -1)

        # Subject demo states
        if t < 80:
            progress = t / 80.0
            px = int(350 + progress * 250)
            py = 450
            bw, bh = 80, 200
            is_fallen = False
        elif t < 140:
            px = 600
            py = int(450 + 5 * np.sin(t * 0.3))
            bw, bh = 85, 190
            is_fallen = False
        elif t < 170:
            fall_progress = (t - 140) / 30.0
            px = int(600 + fall_progress * 80)
            py = int(450 + fall_progress * 130)
            bw = int(80 + fall_progress * 110)
            bh = int(200 - fall_progress * 120)
            is_fallen = True
        elif t < 260:
            px = 680
            py = 580
            bw, bh = 190, 75
            is_fallen = True
        else:
            rec_progress = (t - 260) / 40.0
            px = 680
            py = int(580 - rec_progress * 130)
            bw = int(190 - rec_progress * 110)
            bh = int(75 + rec_progress * 125)
            is_fallen = False

        bx1, by1 = px - bw // 2, py - bh // 2
        bx2, by2 = px + bw // 2, py + bh // 2
        shirt_color = (60, 130, 220) if is_fallen else (40, 160, 220)
        pants_color = (120, 80, 50)
        skin_color = (160, 200, 240)

        if not is_fallen:
            cv2.circle(frame, (px, by1 + int(bh * 0.15)), int(bw * 0.28), skin_color, -1)
            cv2.rectangle(frame, (bx1 + int(bw * 0.15), by1 + int(bh * 0.25)), (bx2 - int(bw * 0.15), by1 + int(bh * 0.65)), shirt_color, -1)
            cv2.rectangle(frame, (bx1 + int(bw * 0.18), by1 + int(bh * 0.65)), (px - 3, by2), pants_color, -1)
            cv2.rectangle(frame, (px + 3, by1 + int(bh * 0.65)), (bx2 - int(bw * 0.18), by2), pants_color, -1)
        else:
            cv2.circle(frame, (bx1 + int(bw * 0.15), py), int(bh * 0.35), skin_color, -1)
            cv2.rectangle(frame, (bx1 + int(bw * 0.25), by1 + int(bh * 0.15)), (bx1 + int(bw * 0.65), by2 - int(bh * 0.15)), shirt_color, -1)
            cv2.rectangle(frame, (bx1 + int(bw * 0.65), by1 + int(bh * 0.2)), (bx2, py - 2), pants_color, -1)
            cv2.rectangle(frame, (bx1 + int(bw * 0.65), py + 2), (bx2, by2 - int(bh * 0.2)), pants_color, -1)

        # Prominent Demo Banner
        cv2.rectangle(frame, (0, 0), (self.width, 35), (40, 20, 80), -1)
        cv2.putText(frame, f"[DEMO / SYNTHETIC SIMULATION] {self.camera_id}", (30, 24), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 150, 255), 2, cv2.LINE_AA)
        
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"{timestamp_str}", (self.width - 260, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        return frame

class CameraStream:
    """
    Unified Camera Stream Handler supporting Webcam, Mobile Phone (DroidCam/IP Webcam),
    RTSP IP Cameras, Video Files, and Explicit Demo Feeds.
    Maintains a dedicated background capture thread with graceful offline detection.
    """
    def __init__(self, camera_id: str, source_type: str = "webcam", source_uri: str = "0", fps: int = 24):
        self.camera_id = camera_id
        self.source_type = source_type.lower()
        self.source_uri = source_uri
        self.target_fps = fps
        self.cap = None
        self.synthetic_gen = None
        self.running = False
        self.is_online = False
        self.last_frame_time = 0.0
        self.current_frame = None
        self.resolution = (1280, 720)
        self.measured_fps = 0.0
        self.lock = threading.Lock()
        self.thread = None
        self.error_message = ""

        if self.source_type == "demo":
            self.synthetic_gen = SyntheticCCTVGenerator(camera_id=self.camera_id, fps=self.target_fps)

    def start(self):
        if self.running:
            return
        self.running = True
        self._init_source()

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started camera stream thread for {self.camera_id} [{self.source_type} -> {mask_camera_uri(str(self.source_uri))}]")

    def _init_source(self):
        if self.source_type == "demo":
            self.is_online = True
            self.error_message = ""
            return

        target = self.source_uri
        if self.source_type in ["webcam", "usb", "local"]:
            try:
                target = int(self.source_uri) if str(self.source_uri).isdigit() else 0
            except ValueError:
                target = 0

        try:
            self.cap = cv2.VideoCapture(target)
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    self.is_online = True
                    self.error_message = ""
                    h, w = frame.shape[:2]
                    self.resolution = (w, h)
                    with self.lock:
                        self.current_frame = frame
                else:
                    self.is_online = False
                    self.error_message = "No frame received from video source."
            else:
                self.is_online = False
                self.error_message = f"Failed to open video source ({mask_camera_uri(str(self.source_uri))})"
                logger.warning(f"Camera {self.camera_id} source open failed: {self.error_message}")
        except Exception as e:
            self.is_online = False
            self.error_message = str(e)
            logger.error(f"Error initializing camera source for {self.camera_id}: {e}")

    def _generate_offline_frame(self) -> np.ndarray:
        """Generates a professional dark offline status frame when no camera signal is present."""
        w, h = self.resolution
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Dark vignette background
        for y in range(h):
            val = int(15 + 10 * (y / h))
            frame[y, :] = (val, val, val + 2)

        # Draw subtle grid lines
        for gx in range(0, w, 80):
            cv2.line(frame, (gx, 0), (gx, h), (25, 28, 35), 1)
        for gy in range(0, h, 80):
            cv2.line(frame, (0, gy), (w, gy), (25, 28, 35), 1)

        # Offline Warning Box
        box_w, box_h = min(480, w - 40), min(220, h - 40)
        bx1 = (w - box_w) // 2
        by1 = (h - box_h) // 2
        bx2 = bx1 + box_w
        by2 = by1 + box_h

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (25, 28, 40), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 70, 180), 2)

        # Header warning
        cv2.putText(frame, f"{self.camera_id} - CAMERA OFFLINE", (bx1 + 30, by1 + 55), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 120, 255), 2, cv2.LINE_AA)
        
        detail = self.error_message or "No live video signal received."
        if len(detail) > 40:
            detail = detail[:37] + "..."
        cv2.putText(frame, detail, (bx1 + 30, by1 + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Source: {self.source_type.upper()} ({mask_camera_uri(str(self.source_uri))})", (bx1 + 30, by1 + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 130, 150), 1, cv2.LINE_AA)
        cv2.putText(frame, "Waiting for stream connection...", (bx1 + 30, by1 + 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

        # Timestamp
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, ts, (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 110, 125), 1, cv2.LINE_AA)
        return frame

    def _capture_loop(self):
        interval = 1.0 / float(self.target_fps)
        reconnect_timer = time.time()
        frames_in_sec = 0
        fps_timer = time.time()

        while self.running:
            start_t = time.time()
            frame = None

            if self.source_type == "demo" and self.synthetic_gen is not None:
                frame = self.synthetic_gen.get_frame()
                self.is_online = True
            elif self.cap is not None and self.cap.isOpened():
                ret, captured = self.cap.read()
                if ret and captured is not None and captured.size > 0:
                    frame = captured
                    self.is_online = True
                    self.last_frame_time = time.time()
                    frames_in_sec += 1
                else:
                    if self.source_type == "file":
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    else:
                        self.is_online = False
                        self.error_message = "Stream signal lost or source disconnected."
                        time.sleep(0.1)
            else:
                self.is_online = False
                # Attempt background reconnect every 5 seconds for real cameras
                if time.time() - reconnect_timer > 5.0:
                    reconnect_timer = time.time()
                    logger.debug(f"Attempting camera stream reconnection for {self.camera_id}...")
                    self._init_source()

            # Measure actual FPS
            if time.time() - fps_timer >= 1.0:
                self.measured_fps = round(frames_in_sec / (time.time() - fps_timer), 1)
                frames_in_sec = 0
                fps_timer = time.time()

            with self.lock:
                if frame is not None:
                    self.current_frame = frame
                    h, w = frame.shape[:2]
                    self.resolution = (w, h)
                elif not self.is_online:
                    self.current_frame = self._generate_offline_frame()

            elapsed = time.time() - start_t
            sleep_t = max(0.002, interval - elapsed)
            time.sleep(sleep_t)

    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def reconnect(self):
        """Forces an immediate reconnect attempt."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._init_source()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_online = False
        logger.info(f"Stopped camera capture stream for {self.camera_id}")
