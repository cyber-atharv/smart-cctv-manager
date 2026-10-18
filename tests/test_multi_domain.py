import pytest
import numpy as np
import cv2
import time
from src.safety.ppe_detector import PPEDetector
from src.zones.zone_manager import Zone, ZoneManager
from src.tracking.tracker import TrackedPerson

def test_ppe_detector_with_helmet_and_vest():
    detector = PPEDetector(min_helmet_ratio=0.05, min_vest_ratio=0.08)

    # Create dummy person frame (100x200)
    frame = np.zeros((200, 100, 3), dtype=np.uint8)

    # Draw Yellow Hard Hat in head region (top 20%)
    # Yellow in BGR: (0, 255, 255)
    cv2.rectangle(frame, (20, 0), (80, 35), (0, 255, 255), -1)

    # Draw Fluorescent Lime Safety Vest in torso region (y: 45 to 120)
    # Bright Lime in BGR: (0, 255, 128)
    cv2.rectangle(frame, (15, 45), (85, 120), (0, 255, 128), -1)

    result = detector.detect_ppe(frame, bbox=[0, 0, 100, 200])

    assert result["has_helmet"] is True
    assert result["has_vest"] is True
    assert result["is_compliant"] is True
    assert len(result["violations"]) == 0

def test_ppe_detector_missing_gear():
    detector = PPEDetector(min_helmet_ratio=0.05, min_vest_ratio=0.08)

    # Dark background / no PPE
    frame = np.full((200, 100, 3), 40, dtype=np.uint8)

    result = detector.detect_ppe(frame, bbox=[0, 0, 100, 200])

    assert result["has_helmet"] is False
    assert result["has_vest"] is False
    assert result["is_compliant"] is False
    assert len(result["violations"]) == 2

def test_zone_table_occupancy_and_turnover():
    # Define a table zone on normalized [0.2, 0.2] to [0.8, 0.8]
    zone = Zone(
        zone_id="T1",
        camera_id="CAM-01",
        name="Table 1",
        zone_type="table",
        polygon=[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
    )

    frame_shape = (1000, 1000, 3)

    # 1. Empty state
    zone.update([], frame_shape)
    assert zone.occupied is False
    assert zone.occupant_count == 0
    assert zone.total_visits == 0

    # 2. Person sits at table (feet inside zone: x=500, y=500)
    person = TrackedPerson(track_id=1, camera_id="CAM-01", bbox=[400, 300, 600, 500], confidence=0.9)
    zone.update([person], frame_shape)

    assert zone.occupied is True
    assert zone.occupant_count == 1
    assert 1 in zone.occupant_track_ids

    # Simulate 5 seconds dwell
    zone.first_occupied_time = time.time() - 5.0
    zone.update([person], frame_shape)
    assert zone.current_dwell_sec >= 4.5

    # 3. Person leaves table -> Turnover completed
    zone.update([], frame_shape)
    assert zone.occupied is False
    assert zone.total_visits == 1
    assert len(zone.completed_dwell_durations) == 1

def test_zone_queue_wait_time_estimation():
    zone = Zone(
        zone_id="Q1",
        camera_id="CAM-01",
        name="Checkout Queue",
        zone_type="queue",
        polygon=[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
    )
    frame_shape = (500, 500, 3)

    p1 = TrackedPerson(track_id=1, camera_id="CAM-01", bbox=[100, 100, 150, 200], confidence=0.9)
    p2 = TrackedPerson(track_id=2, camera_id="CAM-01", bbox=[200, 100, 250, 200], confidence=0.9)
    p3 = TrackedPerson(track_id=3, camera_id="CAM-01", bbox=[300, 100, 350, 200], confidence=0.9)

    zone.update([p1, p2, p3], frame_shape)
    d = zone.to_dict()

    assert d["occupant_count"] == 3
    # 3 people * 1.5 min = 4.5 min
    assert d["estimated_wait_min"] == 4.5

def test_zone_restricted_intrusion_alert():
    zone = Zone(
        zone_id="R1",
        camera_id="CAM-01",
        name="Restricted Area",
        zone_type="restricted",
        polygon=[[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]
    )
    frame_shape = (400, 400, 3)

    # Person inside restricted area
    intruder = TrackedPerson(track_id=7, camera_id="CAM-01", bbox=[50, 50, 100, 120], confidence=0.95)
    zone.update([intruder], frame_shape)

    assert zone.is_alert is True
    assert "INTRUSION" in zone.alert_message

def test_zone_manager_summary_aggregation():
    zm = ZoneManager(camera_id="CAM-TEST")
    summary = zm.get_summary()

    assert "hospitality" in summary
    assert "queue_analytics" in summary
    assert "parking" in summary
    assert "security" in summary
    assert "zones" in summary
