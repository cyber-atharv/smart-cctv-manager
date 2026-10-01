import pytest
from src.activity.classifier import ActivityClassifier

def test_classifier_prediction():
    clf = ActivityClassifier()
    
    # Test standing feature dict
    standing_feats = {
        "body_angle": 5.0,
        "shoulder_angle": 3.0,
        "hip_angle": 3.0,
        "knee_angle_left": 172.0,
        "knee_angle_right": 172.0,
        "head_angle": 4.0,
        "body_height": 0.65,
        "body_width": 0.22,
        "height_width_ratio": 2.95,
        "vertical_velocity": 0.001,
        "horizontal_velocity": 0.001,
        "movement_speed": 0.005,
        "bbox_area": 0.14,
        "landmark_visibility": 0.95,
        "hip_center_y": 0.55,
        "head_to_ankle_y_diff": 0.60
    }
    
    pred_class, conf, probs = clf.predict(standing_feats)
    assert pred_class in ["standing", "working"]
    assert conf > 0.5
    assert isinstance(probs, dict)

    # Test falling feature dict
    falling_feats = {
        "body_angle": 75.0, # highly horizontal
        "shoulder_angle": 35.0,
        "hip_angle": 30.0,
        "knee_angle_left": 110.0,
        "knee_angle_right": 110.0,
        "head_angle": 45.0,
        "body_height": 0.22,
        "body_width": 0.55,
        "height_width_ratio": 0.40, # dropped aspect ratio
        "vertical_velocity": 0.09,
        "horizontal_velocity": 0.04,
        "movement_speed": 0.10,
        "bbox_area": 0.12,
        "landmark_visibility": 0.85,
        "hip_center_y": 0.85,
        "head_to_ankle_y_diff": 0.15
    }

    pred_class_fall, conf_fall, _ = clf.predict(falling_feats)
    assert pred_class_fall == "falling"
    assert conf_fall >= 0.70
