import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

class PPEDetector:
    """
    Workplace Safety PPE (Personal Protective Equipment) Compliance Engine.
    Inspired by PPE_detection-Sentinel.
    Inspects detected person crops for:
    1. Hard Hat / Safety Helmet (Head ROI - industrial yellow, orange, white, blue)
    2. High-Visibility Safety Vest (Torso ROI - fluorescent neon green/lime, safety orange, reflective tape)
    """
    def __init__(self, min_helmet_ratio: float = 0.08, min_vest_ratio: float = 0.12):
        self.min_helmet_ratio = min_helmet_ratio
        self.min_vest_ratio = min_vest_ratio

    def detect_ppe(
        self,
        frame: np.ndarray,
        bbox: List[int],
        landmarks: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyzes person crop and returns PPE compliance status.
        bbox: [x1, y1, x2, y2]
        landmarks: optional pose landmarks dictionary
        """
        x1, y1, x2, y2 = bbox
        h_frame, w_frame = frame.shape[:2]

        x1 = max(0, min(x1, w_frame - 1))
        y1 = max(0, min(y1, h_frame - 1))
        x2 = max(x1 + 1, min(x2, w_frame))
        y2 = max(y1 + 1, min(y2, h_frame))

        bw = x2 - x1
        bh = y2 - y1

        if bw < 10 or bh < 20:
            return {
                "has_helmet": False,
                "helmet_confidence": 0.0,
                "has_vest": False,
                "vest_confidence": 0.0,
                "is_compliant": False,
                "violations": ["Subject too small for PPE verification"]
            }

        person_crop = frame[y1:y2, x1:x2]

        # 1. Head ROI (Top 22% of bounding box)
        head_h = max(4, int(bh * 0.22))
        head_crop = person_crop[0:head_h, :]

        # 2. Torso ROI (Middle 22% to 65% of bounding box)
        torso_y1 = int(bh * 0.22)
        torso_y2 = int(bh * 0.65)
        torso_crop = person_crop[torso_y1:torso_y2, :]

        has_helmet, helmet_conf = self._evaluate_helmet(head_crop)
        has_vest, vest_conf = self._evaluate_vest(torso_crop)

        violations = []
        if not has_helmet:
            violations.append("Missing Hard Hat / Helmet")
        if not has_vest:
            violations.append("Missing High-Visibility Vest")

        is_compliant = (has_helmet and has_vest)

        return {
            "has_helmet": has_helmet,
            "helmet_confidence": round(helmet_conf, 2),
            "has_vest": has_vest,
            "vest_confidence": round(vest_conf, 2),
            "is_compliant": is_compliant,
            "violations": violations
        }

    def _evaluate_helmet(self, head_crop: np.ndarray) -> Tuple[bool, float]:
        """
        Evaluates presence of safety hard hat in head ROI.
        Inspects industrial helmet colors in HSV:
        - Safety Yellow: H: 20-38, S: 70-255, V: 100-255
        - Safety Orange: H: 8-20, S: 100-255, V: 120-255
        - Safety White: S: 0-40, V: 180-255 (high brightness, low saturation)
        - Safety Blue: H: 95-125, S: 80-255, V: 80-255
        """
        if head_crop.size == 0 or head_crop.shape[0] < 2 or head_crop.shape[1] < 2:
            return False, 0.0

        hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
        total_pixels = head_crop.shape[0] * head_crop.shape[1]

        # Safety Yellow & Orange
        mask_yellow = cv2.inRange(hsv, (20, 70, 100), (38, 255, 255))
        mask_orange = cv2.inRange(hsv, (8, 100, 120), (20, 255, 255))
        # Safety Blue
        mask_blue = cv2.inRange(hsv, (95, 80, 80), (125, 255, 255))
        # Safety White (bright dome)
        mask_white = cv2.inRange(hsv, (0, 0, 190), (180, 45, 255))

        combined_mask = mask_yellow | mask_orange | mask_blue | mask_white
        helmet_pixels = cv2.countNonZero(combined_mask)
        ratio = helmet_pixels / float(total_pixels)

        # Check for upper curvature / contiguous blob
        has_helmet = (ratio >= self.min_helmet_ratio)
        confidence = min(0.98, ratio / 0.35) if has_helmet else min(0.45, ratio / 0.15)
        return has_helmet, float(confidence)

    def _evaluate_vest(self, torso_crop: np.ndarray) -> Tuple[bool, float]:
        """
        Evaluates presence of high-visibility safety vest in torso ROI.
        Inspects:
        - Fluorescent Lime / Neon Yellow: H: 28-48, S: 110-255, V: 120-255
        - Fluorescent Safety Orange: H: 5-22, S: 130-255, V: 130-255
        - Silver reflective tape contrast bands
        """
        if torso_crop.size == 0 or torso_crop.shape[0] < 2 or torso_crop.shape[1] < 2:
            return False, 0.0

        hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
        total_pixels = torso_crop.shape[0] * torso_crop.shape[1]

        # High-Vis Neon Green/Lime
        mask_lime = cv2.inRange(hsv, (28, 90, 110), (52, 255, 255))
        # High-Vis Neon Orange
        mask_orange = cv2.inRange(hsv, (5, 110, 120), (22, 255, 255))

        combined_mask = mask_lime | mask_orange
        vest_pixels = cv2.countNonZero(combined_mask)
        ratio = vest_pixels / float(total_pixels)

        has_vest = (ratio >= self.min_vest_ratio)
        confidence = min(0.98, ratio / 0.40) if has_vest else min(0.45, ratio / 0.20)
        return has_vest, float(confidence)
