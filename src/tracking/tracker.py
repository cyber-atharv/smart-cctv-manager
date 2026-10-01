import numpy as np
import time
from typing import List, Dict, Any, Optional
from collections import deque
import logging

logger = logging.getLogger("cctv.tracking")

def compute_iou(boxA: List[int], boxB: List[int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return max(0.0, min(1.0, iou))

class TrackedPerson:
    def __init__(self, track_id: int, camera_id: str, bbox: List[int], confidence: float):
        self.track_id = track_id
        self.camera_id = camera_id
        self.bbox = bbox
        self.confidence = confidence
        self.first_seen = time.time()
        self.last_seen = time.time()
        
        # Centroid and Dynamics
        x1, y1, x2, y2 = bbox
        self.center_x = (x1 + x2) / 2.0
        self.center_y = (y1 + y2) / 2.0
        self.velocity = (0.0, 0.0) # (vx, vy) normalized/pixels per frame
        self.speed = 0.0
        
        # Temporal buffers
        self.history_len = 30
        self.bbox_history = deque(maxlen=self.history_len)
        self.centroid_history = deque(maxlen=self.history_len)
        self.pose_history = deque(maxlen=self.history_len)
        self.activity_history = deque(maxlen=self.history_len)
        self.fall_state_history = deque(maxlen=self.history_len)
        
        self.bbox_history.append(bbox)
        self.centroid_history.append((self.center_x, self.center_y))

        # Identity Layer (Decoupled from Safety)
        self.employee_id: Optional[int] = None
        self.employee_code: Optional[str] = None
        self.employee_name: Optional[str] = None
        self.identity_confidence: float = 0.0
        self.identity_status: str = "unknown" # "recognized", "unknown", "pending"
        
        # State tracking
        self.current_activity: str = "standing"
        self.activity_confidence: float = 0.0
        self.fall_stage: str = "NORMAL" # NORMAL, SUSPICIOUS, POSSIBLE_FALL, CONFIRMED_FALL
        self.fall_confirmed_timestamp: Optional[float] = None
        self.last_event_dispatched_at: float = 0.0
        self.age_frames = 1
        self.time_since_update = 0

    def update(self, bbox: List[int], confidence: float):
        self.last_seen = time.time()
        self.age_frames += 1
        self.time_since_update = 0
        self.confidence = confidence
        
        # Smooth bbox (Exponential moving average)
        alpha = 0.7
        self.bbox = [
            int(alpha * bbox[0] + (1 - alpha) * self.bbox[0]),
            int(alpha * bbox[1] + (1 - alpha) * self.bbox[1]),
            int(alpha * bbox[2] + (1 - alpha) * self.bbox[2]),
            int(alpha * bbox[3] + (1 - alpha) * self.bbox[3]),
        ]
        
        prev_cx, prev_cy = self.center_x, self.center_y
        self.center_x = (self.bbox[0] + self.bbox[2]) / 2.0
        self.center_y = (self.bbox[1] + self.bbox[3]) / 2.0
        
        vx = self.center_x - prev_cx
        vy = self.center_y - prev_cy
        self.velocity = (vx, vy)
        self.speed = float(np.sqrt(vx**2 + vy**2))
        
        self.bbox_history.append(self.bbox)
        self.centroid_history.append((self.center_x, self.center_y))

    def set_identity(self, emp_id: Optional[int], emp_code: Optional[str], emp_name: Optional[str], confidence: float):
        if emp_id is not None and confidence > self.identity_confidence:
            self.employee_id = emp_id
            self.employee_code = emp_code
            self.employee_name = emp_name
            self.identity_confidence = confidence
            self.identity_status = "recognized"
        elif self.employee_id is None:
            self.identity_status = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "camera_id": self.camera_id,
            "bbox": self.bbox,
            "center": [round(self.center_x, 1), round(self.center_y, 1)],
            "velocity": [round(self.velocity[0], 2), round(self.velocity[1], 2)],
            "speed": round(self.speed, 2),
            "activity": self.current_activity,
            "activity_confidence": round(self.activity_confidence, 2),
            "fall_stage": self.fall_stage,
            "employee_id": self.employee_id,
            "employee_code": self.employee_code,
            "employee_name": self.employee_name or "Unknown",
            "identity_status": self.identity_status,
            "identity_confidence": round(self.identity_confidence, 2),
            "age_frames": self.age_frames
        }

class MultiObjectTracker:
    """
    Multi-Object Tracker providing persistent Track IDs, velocity tracking,
    and history association across video frames.
    """
    def __init__(self, camera_id: str, iou_threshold: float = 0.3, max_age: int = 25):
        self.camera_id = camera_id
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_track_id = 1
        self.tracks: Dict[int, TrackedPerson] = {}

    def update(self, detections: List[Dict[str, Any]]) -> List[TrackedPerson]:
        """
        Matches incoming person detections with active tracks using IoU & distance metrics.
        """
        # Increment time since update for existing tracks
        for track in self.tracks.values():
            track.time_since_update += 1

        if not detections:
            # Prune old tracks
            self._prune_lost_tracks()
            return [t for t in self.tracks.values() if t.time_since_update == 0]

        active_track_ids = list(self.tracks.keys())
        num_dets = len(detections)
        num_tracks = len(active_track_ids)

        if num_tracks == 0:
            # Initialize all detections as new tracks
            for det in detections:
                t = TrackedPerson(self.next_track_id, self.camera_id, det["bbox"], det["confidence"])
                self.tracks[self.next_track_id] = t
                self.next_track_id += 1
            return list(self.tracks.values())

        # Compute Cost/IoU Matrix
        iou_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
        for i, tid in enumerate(active_track_ids):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = compute_iou(self.tracks[tid].bbox, det["bbox"])

        matched_tracks = set()
        matched_dets = set()

        # Greedy matching by highest IoU above threshold
        while True:
            max_val = -1.0
            best_i, best_j = -1, -1
            for i in range(num_tracks):
                if i in matched_tracks:
                    continue
                for j in range(num_dets):
                    if j in matched_dets:
                        continue
                    if iou_matrix[i, j] > max_val:
                        max_val = iou_matrix[i, j]
                        best_i, best_j = i, j

            if max_val < self.iou_threshold or best_i == -1:
                break

            matched_tracks.add(best_i)
            matched_dets.add(best_j)
            tid = active_track_ids[best_i]
            self.tracks[tid].update(detections[best_j]["bbox"], detections[best_j]["confidence"])

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_dets:
                t = TrackedPerson(self.next_track_id, self.camera_id, det["bbox"], det["confidence"])
                self.tracks[self.next_track_id] = t
                self.next_track_id += 1

        self._prune_lost_tracks()
        return [t for t in self.tracks.values() if t.time_since_update == 0]

    def _prune_lost_tracks(self):
        lost_ids = [tid for tid, t in self.tracks.items() if t.time_since_update > self.max_age]
        for tid in lost_ids:
            del self.tracks[tid]
