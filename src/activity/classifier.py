import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import logging
from src.config import config
from src.features.feature_extractor import FEATURE_COLUMNS, FeatureExtractor

logger = logging.getLogger("cctv.activity")

DEFAULT_CLASSES = ["standing", "sitting", "working", "falling"]

class ActivityClassifier:
    """
    Activity & Fall Detection Classifier with dual mode:
    1. Scikit-Learn Model (Random Forest / Decision Tree) loaded from pickle.
    2. Transparent Geometric Rule-Based Fallback if model file is not yet exported.
    """
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or (config.models_dir / "fall_detection_model.pkl")
        self.metadata_path = config.models_dir / "metadata.json"
        self.columns_path = config.models_dir / "feature_columns.json"
        
        self.model = None
        self.classes = DEFAULT_CLASSES
        self.feature_columns = FEATURE_COLUMNS
        self.model_version = "1.0-fallback"
        self.feature_extractor = FeatureExtractor()

        self.load_model()

    def load_model(self) -> bool:
        """Attempts to load trained ML model and metadata from disk."""
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                
                if self.metadata_path.exists():
                    with open(self.metadata_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        self.classes = meta.get("classes", DEFAULT_CLASSES)
                        self.model_version = meta.get("version", "1.0")
                
                if self.columns_path.exists():
                    with open(self.columns_path, "r", encoding="utf-8") as f:
                        self.feature_columns = json.load(f)

                logger.info(f"Loaded ML Activity Classifier v{self.model_version} with {len(self.classes)} classes.")
                return True
            except Exception as e:
                logger.error(f"Failed to load ML model from {self.model_path}: {e}")
                self.model = None
                return False
        else:
            logger.info(f"No trained model at {self.model_path}. Using geometric decision engine.")
            return False

    def predict(self, feature_dict: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """
        Predicts activity class from extracted features.
        Returns: (predicted_class, confidence, probability_distribution_dict)
        """
        # If ML model is loaded and available
        if self.model is not None:
            try:
                # Ensure features match expected columns
                vec = np.array([[feature_dict.get(c, 0.0) for c in self.feature_columns]], dtype=np.float32)
                
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(vec)[0]
                    cls_probs = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
                    top_idx = int(np.argmax(probs))
                    predicted_class = self.classes[top_idx]
                    confidence = float(probs[top_idx])
                else:
                    pred = self.model.predict(vec)[0]
                    predicted_class = str(pred)
                    confidence = 0.85
                    cls_probs = {c: (0.85 if c == predicted_class else 0.05) for c in self.classes}

                return predicted_class, round(confidence, 3), cls_probs
            except Exception as e:
                logger.warning(f"Error during ML inference ({e}). Falling back to rule-based classification.")

        # Robust Heuristic / Geometric rule-based classification
        return self._heuristic_predict(feature_dict)

    def _heuristic_predict(self, feats: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """
        Transparent rule-based posture classification based on kinematic & geometric features.
        """
        body_angle = feats.get("body_angle", 0.0)
        aspect_ratio = feats.get("height_width_ratio", 2.0)
        knee_angle_l = feats.get("knee_angle_left", 170.0)
        knee_angle_r = feats.get("knee_angle_right", 170.0)
        movement_speed = feats.get("movement_speed", 0.0)
        vert_velocity = feats.get("vertical_velocity", 0.0)

        avg_knee = (knee_angle_l + knee_angle_r) / 2.0

        # Fall detection criteria:
        # High body angle (> 50 deg from vertical), low aspect ratio (< 1.05), or rapid downward descent
        is_horizontal = body_angle >= 50.0 or aspect_ratio < 1.05
        is_dropping = vert_velocity > 0.06 or (body_angle > 45.0 and movement_speed > 0.08)

        if is_horizontal or is_dropping:
            conf = 0.85
            if body_angle > 65.0:
                conf += 0.10
            if aspect_ratio < 0.8:
                conf += 0.05
            conf = min(0.99, conf)
            probs = {"standing": 0.02, "sitting": 0.03, "working": 0.05, "falling": conf}
            return "falling", round(conf, 3), probs

        # Sitting criteria: Knee bent ~ 75-125 deg, aspect ratio lower than standing (~ 1.1 - 1.5)
        if 65.0 <= avg_knee <= 125.0 and aspect_ratio < 1.7:
            conf = 0.82
            probs = {"standing": 0.08, "sitting": conf, "working": 0.08, "falling": 0.02}
            return "sitting", round(conf, 3), probs

        # Working criteria: Active motion / moderate knee flexion / upper body leaning slightly
        if movement_speed > 0.03 or (15.0 <= body_angle <= 40.0 and avg_knee < 160.0):
            conf = 0.78
            probs = {"standing": 0.12, "sitting": 0.08, "working": conf, "falling": 0.02}
            return "working", round(conf, 3), probs

        # Default Standing
        conf = 0.88
        probs = {"standing": conf, "sitting": 0.05, "working": 0.05, "falling": 0.02}
        return "standing", round(conf, 3), probs
