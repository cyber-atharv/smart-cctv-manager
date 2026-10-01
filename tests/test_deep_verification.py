import pytest
import time
import numpy as np
import cv2
from sqlalchemy.orm import Session

from src.database.db import SessionLocal, init_db
from src.database.models import Employee, FaceProfile, Event
from src.face.embedder import TorchVisionDeepEmbedder, SpatialTextureFaceEmbedder, BaseFaceEmbedder
from src.face.recognizer import FaceRecognizer
from src.events.clip_manager import RollingVideoBuffer, ClipManager
from src.events.event_manager import EventManager
from src.tracking.tracker import MultiObjectTracker
from src.activity.temporal_verifier import TemporalFallVerifier, FallVerificationState

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()

def test_face_embedder_discrimination():
    """
    Verifies that face embedder distinguishes different facial features
    and produces high cosine similarity for identical subjects.
    """
    embedder = TorchVisionDeepEmbedder()
    
    # Create two visually distinct face images
    face1 = np.zeros((112, 112, 3), dtype=np.uint8)
    cv2.rectangle(face1, (20, 20), (90, 90), (220, 180, 140), -1) # Light skin block
    cv2.circle(face1, (40, 45), 10, (20, 20, 20), -1)             # Dark eyes
    cv2.circle(face1, (72, 45), 10, (20, 20, 20), -1)

    face2 = np.zeros((112, 112, 3), dtype=np.uint8)
    cv2.rectangle(face2, (10, 10), (100, 100), (100, 70, 40), -1) # Dark skin block
    cv2.rectangle(face2, (30, 40), (82, 55), (200, 200, 200), -1) # Sunglasses/mask

    emb1 = embedder.get_embedding(face1)
    emb2 = embedder.get_embedding(face2)
    emb1_copy = embedder.get_embedding(face1.copy())

    assert emb1 is not None and len(emb1) == 128
    assert emb2 is not None and len(emb2) == 128

    # Self match should be ~1.0
    sim_self = BaseFaceEmbedder.compute_similarity(emb1, emb1_copy)
    assert sim_self > 0.99

    # Cross match between distinct subjects should be significantly lower
    sim_cross = BaseFaceEmbedder.compute_similarity(emb1, emb2)
    assert sim_cross < sim_self

def test_unknown_person_fall_safety_decoupling():
    """
    Tests:
    Unknown person + Fall -> Valid safety event dispatched.
    Proves that identity recognition is never a blocker for safety alerts.
    """
    db = SessionLocal()
    try:
        event_mgr = EventManager()
        
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "TEST UNKNOWN FALL", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        evt = event_mgr.create_event(
            db=db,
            camera_id="CAM-TEST",
            track_id=42,
            employee_id=None, # UNKNOWN SUBJECT
            activity="falling",
            activity_confidence=0.95,
            identity_confidence=0.0,
            identity_status="unknown",
            current_frame=dummy_frame,
            notes="Unknown visitor fall test"
        )

        assert evt.id.startswith("EVT-")
        assert evt.employee_id is None
        assert evt.identity_status == "unknown"
        assert evt.activity == "falling"
        assert evt.status == "pending_review"
        assert evt.snapshot_path != ""

        reviewed = event_mgr.review_event(db, evt.id, decision="confirmed", note="Confirmed by supervisor")
        assert reviewed.status == "confirmed"
    finally:
        db.close()

def test_multi_person_simultaneous_tracking():
    """
    Tests:
    Multiple people in scene simultaneously (Track 1 standing, Track 2 falling).
    Verifies that tracker and temporal verifier maintain isolated states.
    """
    tracker = MultiObjectTracker(camera_id="CAM-01", iou_threshold=0.3)
    verifier = TemporalFallVerifier(
        suspicious_frames_threshold=2,
        possible_fall_frames_threshold=4,
        floor_persistence_threshold=5,
        fall_confidence_threshold=0.70
    )

    dets_f1 = [
        {"bbox": [100, 100, 180, 300], "confidence": 0.9, "class_name": "person"},
        {"bbox": [400, 100, 480, 300], "confidence": 0.92, "class_name": "person"}
    ]
    tracks = tracker.update(dets_f1)
    assert len(tracks) == 2
    t1_id = tracks[0].track_id
    t2_id = tracks[1].track_id

    # Person 1 stands (Normal)
    stage1, trig1, _ = verifier.process_frame(t1_id, "standing", 0.92, {"body_angle": 5.0, "height_width_ratio": 2.5})
    
    # Person 2 falls (Falling candidate loop)
    triggered_at_least_once = False
    for frame_idx in range(6):
        stage2, trig2, _ = verifier.process_frame(t2_id, "falling", 0.94, {"body_angle": 75.0, "height_width_ratio": 0.5})
        if trig2:
            triggered_at_least_once = True

    # Track 1 remains Normal
    assert stage1 == FallVerificationState.NORMAL
    assert not trig1

    # Track 2 reached confirmed fall and dispatched trigger
    assert stage2 == FallVerificationState.CONFIRMED_FALL
    assert triggered_at_least_once is True

def test_rolling_video_buffer_pre_and_post_seconds():
    """
    Tests:
    RollingVideoBuffer stores circular frame history and extracts pre/post window.
    """
    fps = 10
    buffer = RollingVideoBuffer(fps=fps, max_seconds=5)
    
    t_start = time.time()
    for i in range(25):
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        buffer.add_frame(dummy)
        time.sleep(0.01)

    all_frames = buffer.get_all()
    assert len(all_frames) > 0
    window = buffer.get_window(t_start, time.time())
    assert len(window) == len(all_frames)
