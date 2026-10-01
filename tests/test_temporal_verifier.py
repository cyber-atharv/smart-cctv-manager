import pytest
from src.activity.temporal_verifier import TemporalFallVerifier, FallVerificationState

def test_temporal_verifier_state_machine():
    verifier = TemporalFallVerifier(
        suspicious_frames_threshold=2,
        possible_fall_frames_threshold=4,
        floor_persistence_threshold=6,
        fall_confidence_threshold=0.70,
        debounce_seconds=5.0
    )

    track_id = 101
    fall_features = {"body_angle": 75.0, "height_width_ratio": 0.5}

    # Frame 1: Normal initially
    stage, trigger, _ = verifier.process_frame(track_id, "falling", 0.9, fall_features)
    assert stage == FallVerificationState.NORMAL
    assert not trigger

    # Frame 2: Transition to SUSPICIOUS
    stage, trigger, _ = verifier.process_frame(track_id, "falling", 0.9, fall_features)
    assert stage == FallVerificationState.SUSPICIOUS
    assert not trigger

    # Frames 3 & 4: Transition to POSSIBLE_FALL
    verifier.process_frame(track_id, "falling", 0.9, fall_features)
    stage, trigger, _ = verifier.process_frame(track_id, "falling", 0.9, fall_features)
    assert stage == FallVerificationState.POSSIBLE_FALL
    assert not trigger

    # Frames 5 & 6: Transition to CONFIRMED_FALL with alarm trigger
    verifier.process_frame(track_id, "falling", 0.9, fall_features)
    stage, trigger, conf = verifier.process_frame(track_id, "falling", 0.9, fall_features)
    assert stage == FallVerificationState.CONFIRMED_FALL
    assert trigger is True # Trigger alarm on confirmed state
