import cv2
import time
import os
import threading
from pathlib import Path
from typing import List, Tuple, Optional, Any
from collections import deque
import logging
from src.config import config

logger = logging.getLogger("cctv.clip_manager")

class RollingVideoBuffer:
    """
    In-memory circular frame buffer per camera.
    Stores timestamped (frame, timestamp) tuples to allow retrospective video clip extraction.
    """
    def __init__(self, fps: int = 15, max_seconds: int = 20):
        self.fps = fps
        self.max_frames = fps * max_seconds
        self.buffer = deque(maxlen=self.max_frames)
        self.lock = threading.Lock()

    def add_frame(self, frame: Any):
        with self.lock:
            # Store copy to avoid concurrent modification issues
            self.buffer.append((frame.copy(), time.time()))

    def get_window(self, start_time: float, end_time: float) -> List[Tuple[Any, float]]:
        with self.lock:
            return [item for item in self.buffer if start_time <= item[1] <= end_time]

    def get_all(self) -> List[Tuple[Any, float]]:
        with self.lock:
            return list(self.buffer)

class ClipManager:
    """
    Manages snapshot capture and asynchronous short video clip compilation
    for safety incidents (e.g. 5s pre-event + 10s post-event).
    """
    def __init__(self):
        self.snapshots_dir = config.snapshots_dir
        self.clips_dir = config.clips_dir

    def save_snapshot(self, frame: Any, event_id: str) -> str:
        """Saves current frame as an event snapshot JPEG."""
        if frame is None or frame.size == 0:
            return ""

        filename = f"{event_id}_snapshot.jpg"
        filepath = self.snapshots_dir / filename
        try:
            # Draw timestamp / incident overlay
            annotated = frame.copy()
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                annotated,
                f"SAFETY ALERT | {event_id} | {timestamp_str}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
            cv2.imwrite(str(filepath), annotated)
            logger.info(f"Saved snapshot: {filepath}")
            return f"/storage/snapshots/{filename}"
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")
            return ""

    def save_clip_async(
        self,
        event_id: str,
        buffer: RollingVideoBuffer,
        trigger_time: float,
        pre_seconds: int = 5,
        post_seconds: int = 7,
        fps: int = 15
    ) -> str:
        """
        Schedules asynchronous creation of incident MP4 clip.
        """
        filename = f"{event_id}_clip.mp4"
        relative_path = f"/storage/clips/{filename}"
        target_path = self.clips_dir / filename

        def _recorder():
            # Wait for post_seconds to accumulate in buffer
            time.sleep(post_seconds + 0.5)
            
            frames_with_ts = buffer.get_window(trigger_time - pre_seconds, trigger_time + post_seconds)
            if not frames_with_ts:
                # If window is sparse, fallback to all recent frames
                frames_with_ts = buffer.get_all()

            if not frames_with_ts:
                logger.warning(f"No frames available to write clip for {event_id}")
                return

            h, w = frames_with_ts[0][0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Universal fallback MP4 codec
            out = cv2.VideoWriter(str(target_path), fourcc, fps, (w, h))

            for f, ts in frames_with_ts:
                out.write(f)
            out.release()
            logger.info(f"Successfully rendered event clip: {target_path} ({len(frames_with_ts)} frames)")

        t = threading.Thread(target=_recorder, daemon=True)
        t.start()
        return relative_path
