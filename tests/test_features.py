import pytest
import numpy as np
from src.features.feature_extractor import FeatureExtractor, FEATURE_COLUMNS, calculate_angle_2d

def test_calculate_angle_2d():
    p1 = {"x": 0.0, "y": 1.0}
    p2 = {"x": 0.0, "y": 0.0}
    p3 = {"x": 1.0, "y": 0.0}
    angle = calculate_angle_2d(p1, p2, p3)
    assert abs(angle - 90.0) < 1.0

def test_feature_extractor_standing():
    extractor = FeatureExtractor()
    landmarks = {
        "nose": {"x": 0.5, "y": 0.1, "visibility": 0.95},
        "left_shoulder": {"x": 0.45, "y": 0.2, "visibility": 0.95},
        "right_shoulder": {"x": 0.55, "y": 0.2, "visibility": 0.95},
        "left_hip": {"x": 0.46, "y": 0.55, "visibility": 0.95},
        "right_hip": {"x": 0.54, "y": 0.55, "visibility": 0.95},
        "left_knee": {"x": 0.46, "y": 0.75, "visibility": 0.95},
        "right_knee": {"x": 0.54, "y": 0.75, "visibility": 0.95},
        "left_ankle": {"x": 0.46, "y": 0.95, "visibility": 0.95},
        "right_ankle": {"x": 0.54, "y": 0.95, "visibility": 0.95},
    }
    feats = extractor.extract(landmarks, bbox=[450, 100, 550, 700], frame_shape=(720, 1280))
    
    assert "body_angle" in feats
    assert "height_width_ratio" in feats
    assert feats["body_angle"] < 20.0 # Standing should be nearly upright
    assert feats["height_width_ratio"] > 1.5 # Standing aspect ratio is tall

    vec = extractor.to_feature_vector(feats)
    assert len(vec) == len(FEATURE_COLUMNS)
