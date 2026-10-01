import time
from typing import Dict, Any, Tuple, Optional
from collections import deque
import logging
from src.config import config

logger = logging.getLogger("cctv.fall_verifier")

class FallVerificationState:
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    POSSIBLE_FALL = "POSSIBLE_FALL"
    CONFIRMED_FALL = "CONFIRMED_FALL"

class TemporalFallVerifier:
    """
    Temporal State Machine for Fall Verification.
    Prevents instantaneous false alarms through multi-frame temporal reasoning,
    floor persistence validation, and debounce control.
    """
    def __init__(
        self,
        suspicious_frames_threshold: int = 4,
        possible_fall_frames_threshold: int = 8,
        floor_persistence_threshold: int = 15,
        fall_confidence_threshold: float = 0.70,
        debounce_seconds: float = 10.0
    ):
        self.suspicious_frames_threshold = suspicious_frames_threshold
        self.possible_fall_frames_threshold = possible_fall_frames_threshold
        self.floor_persistence_threshold = floor_persistence_threshold
        self.fall_confidence_threshold = fall_confidence_threshold
        self.debounce_seconds = debounce_seconds

        # Per-track state memory: track_id -> dict
        self.track_states: Dict[int, Dict[str, Any]] = {}

    def _get_or_create_state(self, track_id: int) -> Dict[str, Any]:
        if track_id not in self.track_states:
            self.track_states[track_id] = {
                "state": FallVerificationState.NORMAL,
                "fall_frame_count": 0,
                "floor_persistence_count": 0,
                "recovery_frame_count": 0,
                "first_detected_at": None,
                "confirmed_at": None,
                "last_event_dispatched_at": 0.0,
                "confidence_accumulator": deque(maxlen=20)
            }
        return self.track_states[track_id]

    def process_frame(
        self,
        track_id: int,
        predicted_activity: str,
        activity_confidence: float,
        features: Dict[str, float]
    ) -> Tuple[str, bool, float]:
        """
        Updates temporal state machine for the given track.
        Returns: (current_stage, should_trigger_alarm, aggregated_confidence)
        """
        st = self._get_or_create_state(track_id)
        now = time.time()
        
        is_fall_candidate = (
            predicted_activity == "falling" and activity_confidence >= self.fall_confidence_threshold
        )
        # Check floor persistence / horizontal aspect ratio
        is_near_floor_or_horizontal = (
            features.get("body_angle", 0.0) >= 50.0 or features.get("height_width_ratio", 2.0) <= 1.10
        )

        st["confidence_accumulator"].append(activity_confidence if is_fall_candidate else (1.0 - activity_confidence))

        # State Machine Transitions
        if is_fall_candidate or is_near_floor_or_horizontal:
            st["fall_frame_count"] += 1
            st["recovery_frame_count"] = 0
            
            if is_near_floor_or_horizontal:
                st["floor_persistence_count"] += 1

            if st["state"] == FallVerificationState.NORMAL:
                if st["fall_frame_count"] >= self.suspicious_frames_threshold:
                    st["state"] = FallVerificationState.SUSPICIOUS
                    st["first_detected_at"] = now
                    logger.debug(f"[Track {track_id}] State changed -> SUSPICIOUS")

            elif st["state"] == FallVerificationState.SUSPICIOUS:
                if st["fall_frame_count"] >= self.possible_fall_frames_threshold:
                    st["state"] = FallVerificationState.POSSIBLE_FALL
                    logger.debug(f"[Track {track_id}] State changed -> POSSIBLE_FALL")

            elif st["state"] == FallVerificationState.POSSIBLE_FALL:
                # Require floor persistence to confirm fall and eliminate false drops (e.g. quick shoe tie)
                if st["floor_persistence_count"] >= self.floor_persistence_threshold:
                    st["state"] = FallVerificationState.CONFIRMED_FALL
                    st["confirmed_at"] = now
                    logger.warning(f"[Track {track_id}] State changed -> CONFIRMED_FALL")

        else:
            # Person is standing/sitting/working normally
            st["recovery_frame_count"] += 1
            if st["recovery_frame_count"] >= 10:
                # Reset counts upon sustained recovery
                st["fall_frame_count"] = max(0, st["fall_frame_count"] - 2)
                st["floor_persistence_count"] = max(0, st["floor_persistence_count"] - 2)
                if st["fall_frame_count"] == 0:
                    st["state"] = FallVerificationState.NORMAL

        should_trigger = False
        if st["state"] == FallVerificationState.CONFIRMED_FALL:
            if (now - st["last_event_dispatched_at"]) > self.debounce_seconds:
                should_trigger = True
                st["last_event_dispatched_at"] = now

        avg_conf = float(sum(st["confidence_accumulator"]) / max(1, len(st["confidence_accumulator"])))
        return st["state"], should_trigger, round(avg_conf, 3)

    def cleanup_track(self, track_id: int):
        if track_id in self.track_states:
            del self.track_states[track_id]
