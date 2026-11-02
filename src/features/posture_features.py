"""
features.py — Canonical Feature Definitions for CCTV Posture Detection
Shared across app.py (inference API), train_posture_model.py, tests, and index.html.

All features are scale-invariant biomechanical metrics calculated from
the 33 standard MediaPipe Pose landmarks.
"""

import math
from typing import List, Dict, Any, Union

FEATURE_NAMES = [
    "torso_angle_deg",
    "rel_head_elevation",
    "torso_length_ratio",
    "cog_y",
    "knee_angle_deg",
    "shoulder_tilt_deg",
    "visibility_lower_body",
    "head_shoulder_distance"
]

CLASSES = ["STANDING", "SITTING", "SLEEPING", "LAYING_ON_FLOOR"]

def _vis(lm: Dict[str, Any]) -> float:
    return float(lm.get("visibility", 1.0)) if lm else 0.0

def _calculate_angle_3pt(p1: Dict[str, float], p2: Dict[str, float], p3: Dict[str, float]) -> float:
    """Calculate angle at p2 between vectors (p1-p2) and (p3-p2) in degrees [0, 180]."""
    v1x, v1y = p1["x"] - p2["x"], p1["y"] - p2["y"]
    v2x, v2y = p3["x"] - p2["x"], p3["y"] - p2["y"]
    d1 = math.hypot(v1x, v1y)
    d2 = math.hypot(v2x, v2y)
    if d1 < 1e-4 or d2 < 1e-4:
        return 180.0
    dot = (v1x * v2x + v1y * v2y) / (d1 * d2)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))

def compute_features_from_landmarks(landmarks: List[Dict[str, float]]) -> List[float]:
    """
    Extracts the canonical 8-element feature vector from 33 MediaPipe pose landmarks.
    landmarks: list of 33 dicts with keys 'x', 'y', 'z', and optional 'visibility'.
    """
    if not landmarks or len(landmarks) < 33:
        raise ValueError(f"Expected 33 landmarks, received {len(landmarks) if landmarks else 0}")

    nose = landmarks[0]
    left_shoulder, right_shoulder = landmarks[11], landmarks[12]
    left_hip, right_hip = landmarks[23], landmarks[24]
    left_knee, right_knee = landmarks[25], landmarks[26]
    left_ankle, right_ankle = landmarks[27], landmarks[28]

    # Shoulder width & midpoint
    sh_dx = right_shoulder["x"] - left_shoulder["x"]
    sh_dy = right_shoulder["y"] - left_shoulder["y"]
    shoulder_width = max(0.01, math.hypot(sh_dx, sh_dy))
    mid_shoulder_x = (left_shoulder["x"] + right_shoulder["x"]) / 2.0
    mid_shoulder_y = (left_shoulder["y"] + right_shoulder["y"]) / 2.0

    # Desk occlusion hip synthesis if hips not visible
    hips_visible = (_vis(left_hip) >= 0.45 and _vis(right_hip) >= 0.45)
    if not hips_visible:
        drop_y = mid_shoulder_y + (1.55 * shoulder_width)
        mid_hip_x = mid_shoulder_x
        mid_hip_y = drop_y
    else:
        mid_hip_x = (left_hip["x"] + right_hip["x"]) / 2.0
        mid_hip_y = (left_hip["y"] + right_hip["y"]) / 2.0

    # 1. Torso Angle from vertical (degrees, 0 = upright vertical, ~90 = horizontal)
    dy = mid_hip_y - mid_shoulder_y
    dx = abs(mid_shoulder_x - mid_hip_x)
    torso_angle_deg = math.degrees(math.atan2(dx, max(0.001, abs(dy))))

    # 2. Relative Head Elevation (normalized by shoulder width)
    head_y = nose["y"]
    head_elevation = mid_shoulder_y - head_y  # positive when head is above shoulders
    rel_head_elevation = head_elevation / shoulder_width

    # 3. Torso Length Ratio (torso length / shoulder width)
    torso_len = max(0.01, math.hypot(mid_hip_x - mid_shoulder_x, mid_hip_y - mid_shoulder_y))
    torso_length_ratio = torso_len / shoulder_width

    # 4. Center of Gravity Y (normalized vertical position along torso, 0 top to 1 floor)
    cog_y = max(0.0, min(1.0, mid_shoulder_y + (dy * 0.55)))

    # 5. Knee Angle (degrees)
    left_knee_ok = _vis(left_hip) > 0.4 and _vis(left_knee) > 0.4 and _vis(left_ankle) > 0.4
    right_knee_ok = _vis(right_hip) > 0.4 and _vis(right_knee) > 0.4 and _vis(right_ankle) > 0.4
    
    if left_knee_ok and right_knee_ok:
        knee_angle_deg = (_calculate_angle_3pt(left_hip, left_knee, left_ankle) + 
                          _calculate_angle_3pt(right_hip, right_knee, right_ankle)) / 2.0
    elif left_knee_ok:
        knee_angle_deg = _calculate_angle_3pt(left_hip, left_knee, left_ankle)
    elif right_knee_ok:
        knee_angle_deg = _calculate_angle_3pt(right_hip, right_knee, right_ankle)
    else:
        # Occluded legs (e.g. seated behind desk): default to 90.0°
        knee_angle_deg = 90.0

    # 6. Lateral Shoulder Tilt (degrees from horizontal)
    shoulder_tilt_deg = math.degrees(math.atan2(abs(sh_dy), max(0.001, abs(sh_dx))))

    # 7. Lower Body Visibility (mean visibility of hips, knees, ankles)
    lower_lms = [left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]
    visibility_lower_body = sum(_vis(lm) for lm in lower_lms) / 6.0

    # 8. Head-to-Shoulder Distance relative to torso length
    head_dist = math.hypot(nose["x"] - mid_shoulder_x, nose["y"] - mid_shoulder_y)
    head_shoulder_distance = head_dist / torso_len

    return [
        round(torso_angle_deg, 4),
        round(rel_head_elevation, 4),
        round(torso_length_ratio, 4),
        round(cog_y, 4),
        round(knee_angle_deg, 4),
        round(shoulder_tilt_deg, 4),
        round(visibility_lower_body, 4),
        round(head_shoulder_distance, 4)
    ]

