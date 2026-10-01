import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path
from src.features.feature_extractor import FEATURE_COLUMNS

class DatasetAdapter:
    """
    Abstract adapter interface for normalizing external fall datasets
    (UR Fall, UP-Fall, FallVision, Synthetic Biomechanical Benchmarks)
    into the standard 16-feature schema.
    """
    def transform(self, raw_data: Any) -> pd.DataFrame:
        raise NotImplementedError

class BiomechanicalRealisticGenerator(DatasetAdapter):
    """
    Generates a realistic multi-subject dataset incorporating:
    - Biomechanical kinematics across diverse heights, weights, and camera mounting angles.
    - Borderline & Hard Negative classes (Tying shoes, bending over, sitting down quickly, stumbling without falling).
    - Measurement jitter, sensor noise, and partial landmark occlusions.
    """
    def __init__(self, n_subjects: int = 30, samples_per_subject: int = 200, random_state: int = 42):
        self.n_subjects = n_subjects
        self.samples_per_subject = samples_per_subject
        self.random_state = random_state

    def generate(self) -> pd.DataFrame:
        np.random.seed(self.random_state)
        records = []

        for sub_id in range(1, self.n_subjects + 1):
            # Subject specific anatomical variation
            subject_height_scale = np.random.uniform(0.88, 1.12)
            subject_noise_factor = np.random.uniform(0.02, 0.06)

            for _ in range(self.samples_per_subject):
                # Class selection including realistic non-fall edge cases
                choice = np.random.choice(
                    ["standing", "sitting", "working", "falling", "crouch_tie_shoe", "rapid_sit", "bend_pick_object"],
                    p=[0.25, 0.22, 0.23, 0.15, 0.05, 0.05, 0.05]
                )

                # Base noise
                jitter = lambda: np.random.normal(0, subject_noise_factor)

                if choice == "standing":
                    label = "standing"
                    body_angle = max(0.0, np.random.normal(6.0, 4.0))
                    shoulder_angle = max(0.0, np.random.normal(4.0, 3.0))
                    hip_angle = max(0.0, np.random.normal(4.0, 3.0))
                    knee_l = np.clip(np.random.normal(172.0, 5.0), 150.0, 180.0)
                    knee_r = np.clip(np.random.normal(172.0, 5.0), 150.0, 180.0)
                    head_angle = max(0.0, np.random.normal(5.0, 3.5))
                    body_h = np.clip(0.65 * subject_height_scale + jitter(), 0.45, 0.85)
                    body_w = np.clip(0.22 + jitter(), 0.15, 0.32)
                    v_vert = np.random.normal(0.001, 0.008)
                    v_horiz = np.random.normal(0.002, 0.01)
                    vis = np.clip(np.random.uniform(0.82, 0.98), 0.5, 1.0)
                    hip_y = np.clip(0.55 + jitter(), 0.40, 0.70)
                    head_ank = np.clip(0.60 * subject_height_scale + jitter(), 0.45, 0.80)

                elif choice == "sitting":
                    label = "sitting"
                    body_angle = max(0.0, np.random.normal(14.0, 6.0))
                    shoulder_angle = max(0.0, np.random.normal(5.0, 3.5))
                    hip_angle = max(0.0, np.random.normal(6.0, 4.0))
                    knee_l = np.clip(np.random.normal(92.0, 10.0), 65.0, 125.0)
                    knee_r = np.clip(np.random.normal(92.0, 10.0), 65.0, 125.0)
                    head_angle = max(0.0, np.random.normal(9.0, 4.5))
                    body_h = np.clip(0.42 * subject_height_scale + jitter(), 0.30, 0.55)
                    body_w = np.clip(0.28 + jitter(), 0.20, 0.38)
                    v_vert = np.random.normal(0.00, 0.005)
                    v_horiz = np.random.normal(0.00, 0.005)
                    vis = np.clip(np.random.uniform(0.78, 0.95), 0.5, 1.0)
                    hip_y = np.clip(0.66 + jitter(), 0.55, 0.80)
                    head_ank = np.clip(0.38 * subject_height_scale + jitter(), 0.25, 0.50)

                elif choice == "working":
                    label = "working"
                    body_angle = max(0.0, np.random.normal(26.0, 8.0))
                    shoulder_angle = max(0.0, np.random.normal(12.0, 6.0))
                    hip_angle = max(0.0, np.random.normal(10.0, 5.0))
                    knee_l = np.clip(np.random.normal(148.0, 18.0), 100.0, 175.0)
                    knee_r = np.clip(np.random.normal(148.0, 18.0), 100.0, 175.0)
                    head_angle = max(0.0, np.random.normal(19.0, 7.0))
                    body_h = np.clip(0.58 * subject_height_scale + jitter(), 0.40, 0.75)
                    body_w = np.clip(0.26 + jitter(), 0.18, 0.35)
                    v_vert = np.random.normal(0.015, 0.015)
                    v_horiz = np.random.normal(0.025, 0.02)
                    vis = np.clip(np.random.uniform(0.78, 0.95), 0.5, 1.0)
                    hip_y = np.clip(0.58 + jitter(), 0.45, 0.72)
                    head_ank = np.clip(0.52 * subject_height_scale + jitter(), 0.38, 0.70)

                elif choice == "crouch_tie_shoe":
                    # Hard Negative: Crouching / Tying shoes (low height, high knee bend, overlapping aspect ratio)
                    label = "working"
                    body_angle = max(0.0, np.random.normal(54.0, 14.0)) # Wide variance
                    shoulder_angle = max(0.0, np.random.normal(18.0, 10.0))
                    hip_angle = max(0.0, np.random.normal(14.0, 8.0))
                    knee_l = np.clip(np.random.normal(60.0, 18.0), 30.0, 110.0)
                    knee_r = np.clip(np.random.normal(60.0, 18.0), 30.0, 110.0)
                    head_angle = max(0.0, np.random.normal(38.0, 14.0))
                    body_h = np.clip(0.30 * subject_height_scale + jitter(), 0.18, 0.48)
                    body_w = np.clip(0.38 + jitter(), 0.25, 0.55)
                    v_vert = np.random.normal(0.02, 0.025) # Occasional quick downward movement
                    v_horiz = np.random.normal(0.005, 0.015)
                    vis = np.clip(np.random.uniform(0.65, 0.88), 0.35, 1.0)
                    hip_y = np.clip(0.74 + jitter(), 0.58, 0.90)
                    head_ank = np.clip(0.24 * subject_height_scale + jitter(), 0.12, 0.38)

                elif choice == "bend_pick_object":
                    # Hard Negative: Bending over to pick up an object (temporarily horizontal torso)
                    label = "working"
                    body_angle = max(0.0, np.random.normal(64.0, 12.0)) # Torso strongly bent over
                    shoulder_angle = max(0.0, np.random.normal(24.0, 10.0))
                    hip_angle = max(0.0, np.random.normal(18.0, 9.0))
                    knee_l = np.clip(np.random.normal(155.0, 18.0), 110.0, 180.0)
                    knee_r = np.clip(np.random.normal(155.0, 18.0), 110.0, 180.0)
                    head_angle = max(0.0, np.random.normal(44.0, 15.0))
                    body_h = np.clip(0.36 * subject_height_scale + jitter(), 0.22, 0.52)
                    body_w = np.clip(0.46 + jitter(), 0.30, 0.62)
                    v_vert = np.random.normal(0.035, 0.025) # Moderate downward speed
                    v_horiz = np.random.normal(0.015, 0.015)
                    vis = np.clip(np.random.uniform(0.68, 0.90), 0.4, 1.0)
                    hip_y = np.clip(0.68 + jitter(), 0.50, 0.82)
                    head_ank = np.clip(0.30 * subject_height_scale + jitter(), 0.16, 0.44)

                elif choice == "rapid_sit":
                    # Hard Negative: Fast collapse into a chair
                    label = "sitting"
                    body_angle = max(0.0, np.random.normal(28.0, 10.0))
                    shoulder_angle = max(0.0, np.random.normal(10.0, 6.0))
                    hip_angle = max(0.0, np.random.normal(9.0, 6.0))
                    knee_l = np.clip(np.random.normal(88.0, 14.0), 60.0, 125.0)
                    knee_r = np.clip(np.random.normal(88.0, 14.0), 60.0, 125.0)
                    head_angle = max(0.0, np.random.normal(16.0, 8.0))
                    body_h = np.clip(0.40 * subject_height_scale + jitter(), 0.28, 0.55)
                    body_w = np.clip(0.32 + jitter(), 0.22, 0.45)
                    v_vert = np.random.normal(0.065, 0.03) # High downward speed similar to fall
                    v_horiz = np.random.normal(0.015, 0.015)
                    vis = np.clip(np.random.uniform(0.70, 0.90), 0.4, 1.0)
                    hip_y = np.clip(0.70 + jitter(), 0.55, 0.85)
                    head_ank = np.clip(0.35 * subject_height_scale + jitter(), 0.22, 0.48)

                else: # True Fall
                    label = "falling"
                    body_angle = max(0.0, np.random.normal(70.0, 14.0)) # Torso horizontal
                    shoulder_angle = max(0.0, np.random.normal(32.0, 16.0))
                    hip_angle = max(0.0, np.random.normal(26.0, 14.0))
                    knee_l = np.clip(np.random.normal(118.0, 32.0), 35.0, 175.0)
                    knee_r = np.clip(np.random.normal(118.0, 32.0), 35.0, 175.0)
                    head_angle = max(0.0, np.random.normal(46.0, 18.0))
                    body_h = np.clip(0.26 * subject_height_scale + jitter(), 0.12, 0.42)
                    body_w = np.clip(0.54 + jitter(), 0.35, 0.75)
                    v_vert = np.random.normal(0.075, 0.035) # High descent velocity with overlap
                    v_horiz = np.random.normal(0.04, 0.025)
                    vis = np.clip(np.random.uniform(0.65, 0.85), 0.35, 1.0)
                    hip_y = np.clip(0.81 + jitter(), 0.65, 0.95)
                    head_ank = np.clip(0.20 * subject_height_scale + jitter(), 0.08, 0.36)

                aspect_ratio = body_h / (body_w + 1e-5)
                movement_speed = math.sqrt(v_horiz**2 + v_vert**2)
                bbox_area = body_h * body_w

                records.append({
                    "subject_id": f"SUBJ_{sub_id:02d}",
                    "body_angle": round(float(body_angle), 2),
                    "shoulder_angle": round(float(shoulder_angle), 2),
                    "hip_angle": round(float(hip_angle), 2),
                    "knee_angle_left": round(float(knee_l), 2),
                    "knee_angle_right": round(float(knee_r), 2),
                    "head_angle": round(float(head_angle), 2),
                    "body_height": round(float(body_h), 4),
                    "body_width": round(float(body_w), 4),
                    "height_width_ratio": round(float(aspect_ratio), 4),
                    "vertical_velocity": round(float(v_vert), 4),
                    "horizontal_velocity": round(float(v_horiz), 4),
                    "movement_speed": round(float(movement_speed), 4),
                    "bbox_area": round(float(bbox_area), 4),
                    "landmark_visibility": round(float(vis), 3),
                    "hip_center_y": round(float(hip_y), 4),
                    "head_to_ankle_y_diff": round(float(head_ank), 4),
                    "label": label
                })

        return pd.DataFrame(records)
