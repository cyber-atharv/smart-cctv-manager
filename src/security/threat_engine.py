"""
Universal Smart Vision (USV) - Explainable Threat Scoring Engine
Decomposes composite threat scores (0-100) into explainable, itemized factors.
Department of CSE-AIML, D.Y. Patil College of Engineering & Technology, Kolhapur.
"""

from typing import Dict, Any, List, Optional
import time

class ExplainableThreatEngine:
    """
    Transparent, Rule-Governed Threat Assessment Engine.
    Guarantees no arbitrary black-box scores: every point is tied to an observable factor.
    """

    LOITERING_THRESHOLD_SEC = 12.0
    RESTRICTED_POINTS = 35
    LOITERING_POINTS = 25
    WEAPON_POINTS = 35
    SUSPICIOUS_POSTURE_POINTS = 20
    OFF_HOURS_POINTS = 10
    BASE_PRESENCE_POINTS = 5

    def __init__(self):
        self.active_loitering_timers: Dict[int, float] = {}

    def assess_track(
        self,
        track_id: int,
        is_restricted_zone: bool = False,
        zone_name: str = "",
        stationary_duration: float = 0.0,
        has_weapon_indicator: bool = False,
        weapon_type: str = "",
        fall_stage: str = "NORMAL",
        is_off_hours: bool = False
    ) -> Dict[str, Any]:
        """
        Computes an explainable, decomposed threat score for a tracked entity.
        Returns composite score, category level, and explicit factor list.
        """
        factors: List[Dict[str, Any]] = []
        score = self.BASE_PRESENCE_POINTS
        factors.append({
            "code": "BASE_OBSERVATION",
            "name": "Subject Detected in Field of View",
            "points": self.BASE_PRESENCE_POINTS,
            "severity": "info",
            "detail": f"Track #{track_id} actively monitored by optical sensor."
        })

        # Factor 1: Restricted Zone Intrusion
        if is_restricted_zone:
            score += self.RESTRICTED_POINTS
            factors.append({
                "code": "ZONE_INTRUSION",
                "name": f"Restricted Boundary Trespass: {zone_name or 'Zone'}",
                "points": self.RESTRICTED_POINTS,
                "severity": "danger",
                "detail": f"Subject entered perimeter '{zone_name or 'Restricted Zone'}' without authorized clearance."
            })

        # Factor 2: Suspicious Loitering / Dwell Time
        if stationary_duration >= self.LOITERING_THRESHOLD_SEC:
            loiter_pts = min(self.LOITERING_POINTS, int(15 + (stationary_duration - self.LOITERING_THRESHOLD_SEC)))
            score += loiter_pts
            factors.append({
                "code": "PROLONGED_LOITERING",
                "name": f"Prolonged Stationary Loitering ({round(stationary_duration, 1)}s)",
                "points": loiter_pts,
                "severity": "warning",
                "detail": f"Dwell time exceeds safety protocol threshold of {self.LOITERING_THRESHOLD_SEC}s."
            })

        # Factor 3: Weapon / Concealed Dangerous Object
        if has_weapon_indicator:
            score += self.WEAPON_POINTS
            factors.append({
                "code": "WEAPON_SIGNATURE",
                "name": f"Weapon Anomaly Detected: {weapon_type or 'Suspect Object'}",
                "points": self.WEAPON_POINTS,
                "severity": "danger",
                "detail": f"Computer vision classifier flagged {weapon_type or 'hazardous handheld object'} with high confidence."
            })

        # Factor 4: Postural Distress / Collapse
        if fall_stage in ["CONFIRMED_FALL", "POSSIBLE_FALL"]:
            score += self.SUSPICIOUS_POSTURE_POINTS
            factors.append({
                "code": "POSTURAL_DISTRESS",
                "name": "Subject Postural Collapse / Fall Emergency",
                "points": self.SUSPICIOUS_POSTURE_POINTS,
                "severity": "danger",
                "detail": f"Biomechanical kinematic classifier flagged stage: {fall_stage}."
            })

        # Factor 5: Off-Hours Facility Activity
        if is_off_hours:
            score += self.OFF_HOURS_POINTS
            factors.append({
                "code": "OFF_HOURS",
                "name": "Facility Movement During Inactive Hours",
                "points": self.OFF_HOURS_POINTS,
                "severity": "warning",
                "detail": "Motion registered outside scheduled daylight shift hours."
            })

        final_score = min(100, max(0, score))

        if final_score >= 75:
            level = "CRITICAL"
            badge_color = "var(--danger)"
        elif final_score >= 50:
            level = "ELEVATED"
            badge_color = "#f97316" # Orange
        elif final_score >= 30:
            level = "MODERATE"
            badge_color = "var(--warning)"
        else:
            level = "LOW"
            badge_color = "var(--success)"

        # Formulate human-readable natural language summary
        danger_factors = [f["name"] for f in factors if f["severity"] == "danger"]
        warning_factors = [f["name"] for f in factors if f["severity"] == "warning"]

        if danger_factors:
            summary = f"{level} THREAT: " + "; ".join(danger_factors[:2])
        elif warning_factors:
            summary = f"{level} RISK: " + "; ".join(warning_factors[:2])
        else:
            summary = f"Normal Surveillance: Standard pedestrian observation."

        return {
            "track_id": track_id,
            "threat_score": final_score,
            "threat_level": level,
            "badge_color": badge_color,
            "summary": summary,
            "factors": factors,
            "timestamp": time.time()
        }

threat_engine = ExplainableThreatEngine()

