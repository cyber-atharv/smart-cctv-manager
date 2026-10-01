import math
import numpy as np
from typing import Dict, Any, List, Optional

FEATURE_COLUMNS = [
    "body_angle",
    "shoulder_angle",
    "hip_angle",
    "knee_angle_left",
    "knee_angle_right",
    "head_angle",
    "body_height",
    "body_width",
    "height_width_ratio",
    "vertical_velocity",
    "horizontal_velocity",
    "movement_speed",
    "bbox_area",
    "landmark_visibility",
    "hip_center_y",
    "head_to_ankle_y_diff"
]

def calculate_angle_2d(p1: Dict[str, float], p2: Dict[str, float], p3: Optional[Dict[str, float]] = None) -> float:
    """
    If p3 is given: calculates angle (in degrees) at vertex p2 between (p1->p2) and (p3->p2).
    If p3 is None: calculates inclination angle of segment (p1->p2) relative to vertical/horizontal.
    """
    if p3 is not None:
        v1 = np.array([p1["x"] - p2["x"], p1["y"] - p2["y"]])
        v2 = np.array([p3["x"] - p2["x"], p3["y"] - p2["y"]])
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 < 1e-6 or norm_v2 < 1e-6:
            return 180.0
        cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))
    else:
        # Segment angle from vertical
        dx = p2["x"] - p1["x"]
        dy = p2["y"] - p1["y"]
        angle_rad = math.atan2(abs(dx), abs(dy) + 1e-6)
        return float(math.degrees(angle_rad))

class FeatureExtractor:
    """
    Feature Extractor converting raw body landmarks and tracking history into
    reproducible mathematical feature vectors for the ML activity & fall classifier.
    """
    def __init__(self):
        self.feature_names = FEATURE_COLUMNS

    def extract(
        self,
        landmarks: Dict[str, Dict[str, float]],
        bbox: Optional[List[int]] = None,
        frame_shape: Optional[tuple] = None,
        velocity: Optional[tuple] = None,
        prev_hip_y: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Extracts structured numerical features from landmarks and temporal dynamics.
        """
        feats = {name: 0.0 for name in self.feature_names}
        if not landmarks:
            return feats

        # Key points extraction with fallbacks
        nose = landmarks.get("nose", {"x": 0.5, "y": 0.2, "visibility": 0.5})
        l_sh = landmarks.get("left_shoulder", {"x": 0.45, "y": 0.3, "visibility": 0.5})
        r_sh = landmarks.get("right_shoulder", {"x": 0.55, "y": 0.3, "visibility": 0.5})
        l_hip = landmarks.get("left_hip", {"x": 0.47, "y": 0.6, "visibility": 0.5})
        r_hip = landmarks.get("right_hip", {"x": 0.53, "y": 0.6, "visibility": 0.5})
        l_knee = landmarks.get("left_knee", {"x": 0.47, "y": 0.8, "visibility": 0.5})
        r_knee = landmarks.get("right_knee", {"x": 0.53, "y": 0.8, "visibility": 0.5})
        l_ank = landmarks.get("left_ankle", {"x": 0.47, "y": 0.95, "visibility": 0.5})
        r_ank = landmarks.get("right_ankle", {"x": 0.53, "y": 0.95, "visibility": 0.5})

        mid_sh = {"x": (l_sh["x"] + r_sh["x"]) / 2.0, "y": (l_sh["y"] + r_sh["y"]) / 2.0}
        mid_hip = {"x": (l_hip["x"] + r_hip["x"]) / 2.0, "y": (l_hip["y"] + r_hip["y"]) / 2.0}
        mid_ank = {"x": (l_ank["x"] + r_ank["x"]) / 2.0, "y": (l_ank["y"] + r_ank["y"]) / 2.0}

        # 1. Torso body angle relative to vertical (0 = standing straight, 90 = lying flat)
        dx_torso = abs(mid_sh["x"] - mid_hip["x"])
        dy_torso = abs(mid_sh["y"] - mid_hip["y"])
        feats["body_angle"] = round(math.degrees(math.atan2(dx_torso, dy_torso + 1e-6)), 2)

        # 2. Shoulder line angle relative to horizontal
        dx_sh = abs(r_sh["x"] - l_sh["x"])
        dy_sh = abs(r_sh["y"] - l_sh["y"])
        feats["shoulder_angle"] = round(math.degrees(math.atan2(dy_sh, dx_sh + 1e-6)), 2)

        # 3. Hip line angle relative to horizontal
        dx_hip = abs(r_hip["x"] - l_hip["x"])
        dy_hip = abs(r_hip["y"] - l_hip["y"])
        feats["hip_angle"] = round(math.degrees(math.atan2(dy_hip, dx_hip + 1e-6)), 2)

        # 4. Knee flexion angles (Hip - Knee - Ankle)
        feats["knee_angle_left"] = round(calculate_angle_2d(l_hip, l_knee, l_ank), 2)
        feats["knee_angle_right"] = round(calculate_angle_2d(r_hip, r_knee, r_ank), 2)

        # 5. Head angle from mid-shoulder to nose
        dx_head = abs(nose["x"] - mid_sh["x"])
        dy_head = abs(nose["y"] - mid_sh["y"])
        feats["head_angle"] = round(math.degrees(math.atan2(dx_head, dy_head + 1e-6)), 2)

        # 6. Geometric dimensions & aspect ratio
        all_x = [lm["x"] for lm in landmarks.values()]
        all_y = [lm["y"] for lm in landmarks.values()]
        body_w = max(all_x) - min(all_x)
        body_h = max(all_y) - min(all_y)
        
        feats["body_height"] = round(float(body_h), 4)
        feats["body_width"] = round(float(body_w), 4)
        feats["height_width_ratio"] = round(float(body_h / (body_w + 1e-5)), 4)

        # 7. Velocities & Dynamics
        if velocity is not None:
            vx, vy = velocity
            feats["horizontal_velocity"] = round(float(vx), 4)
            feats["vertical_velocity"] = round(float(vy), 4)
            feats["movement_speed"] = round(float(math.sqrt(vx**2 + vy**2)), 4)
        elif prev_hip_y is not None:
            v_vert = mid_hip["y"] - prev_hip_y
            feats["vertical_velocity"] = round(float(v_vert), 4)
            feats["movement_speed"] = round(abs(float(v_vert)), 4)

        # 8. Bounding box area ratio
        if bbox is not None and frame_shape is not None:
            fh, fw = frame_shape[:2]
            bw = (bbox[2] - bbox[0]) / float(fw)
            bh = (bbox[3] - bbox[1]) / float(fh)
            feats["bbox_area"] = round(float(bw * bh), 4)
            if feats["body_height"] == 0:
                feats["body_height"] = round(float(bh), 4)
                feats["body_width"] = round(float(bw), 4)
                feats["height_width_ratio"] = round(float(bh / (bw + 1e-5)), 4)
        else:
            feats["bbox_area"] = round(float(body_w * body_h), 4)

        # 9. Landmark visibility
        vis_scores = [lm.get("visibility", 0.5) for lm in landmarks.values()]
        feats["landmark_visibility"] = round(float(np.mean(vis_scores)), 3)

        # 10. Floor proximity & head-ankle span
        feats["hip_center_y"] = round(float(mid_hip["y"]), 4)
        feats["head_to_ankle_y_diff"] = round(float(mid_ank["y"] - nose["y"]), 4)

        return feats

    def to_feature_vector(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """Converts feature dict to 1D numpy array ordered strictly by self.feature_names."""
        return np.array([feature_dict.get(col, 0.0) for col in self.feature_names], dtype=np.float32)
