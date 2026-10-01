import pytest
import numpy as np
from src.tracking.tracker import MultiObjectTracker, compute_iou
from src.face.embedder import FaceEmbedder

def test_compute_iou():
    box1 = [10, 10, 100, 100]
    box2 = [10, 10, 100, 100]
    assert compute_iou(box1, box2) == 1.0

    box3 = [200, 200, 300, 300]
    assert compute_iou(box1, box3) == 0.0

def test_tracker_persistence():
    tracker = MultiObjectTracker(camera_id="CAM-TEST", iou_threshold=0.3)
    
    # Frame 1: Initial detection
    dets_f1 = [{"bbox": [100, 100, 200, 300], "confidence": 0.9, "class_name": "person"}]
    tracks_f1 = tracker.update(dets_f1)
    assert len(tracks_f1) == 1
    t_id = tracks_f1[0].track_id

    # Frame 2: Slightly moved detection
    dets_f2 = [{"bbox": [105, 102, 205, 302], "confidence": 0.92, "class_name": "person"}]
    tracks_f2 = tracker.update(dets_f2)
    assert len(tracks_f2) == 1
    assert tracks_f2[0].track_id == t_id # Persistent Track ID verified

def test_face_embedder_normalization():
    embedder = FaceEmbedder()
    dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    emb = embedder.get_embedding(dummy_face)
    
    assert emb is not None
    assert len(emb) == 128
    # L2 norm check ~ 1.0
    norm = np.linalg.norm(np.array(emb))
    assert abs(norm - 1.0) < 0.05
