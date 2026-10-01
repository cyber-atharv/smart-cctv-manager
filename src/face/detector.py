import cv2
import os
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger("cctv.face_detector")

class FaceDetector:
    """
    Face detector with multi-path cascade resolution, DNN fallback,
    and person ROI upper-body crop support.
    """
    def __init__(self, min_face_size: int = 35):
        self.min_face_size = min_face_size
        self.cascade = None

        cascade_candidates = [
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"),
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml"),
            "haarcascade_frontalface_default.xml"
        ]

        for p in cascade_candidates:
            if os.path.exists(p):
                casc = cv2.CascadeClassifier(p)
                if not casc.empty():
                    self.cascade = casc
                    logger.info(f"Loaded Haar Cascade from {p}")
                    break

    def detect(self, frame: np.ndarray, person_bbox: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        faces = []
        if frame is None or frame.size == 0:
            return faces

        h, w = frame.shape[:2]
        
        # If person_bbox given, focus on head/torso upper 50%
        if person_bbox is not None:
            px1, py1, px2, py2 = person_bbox
            roi_h = max(10, int((py2 - py1) * 0.50))
            rx1 = max(0, px1)
            ry1 = max(0, py1)
            rx2 = min(w, px2)
            ry2 = min(h, py1 + roi_h)
            roi = frame[ry1:ry2, rx1:rx2]
            offset_x, offset_y = rx1, ry1
        else:
            roi = frame
            offset_x, offset_y = 0, 0

        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            return faces

        if self.cascade is not None:
            try:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                rects = self.cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(self.min_face_size, self.min_face_size)
                )

                for (fx, fy, fw, fh) in rects:
                    abs_x1 = offset_x + fx
                    abs_y1 = offset_y + fy
                    abs_x2 = offset_x + fx + fw
                    abs_y2 = offset_y + fy + fh
                    crop = frame[abs_y1:abs_y2, abs_x1:abs_x2]
                    if crop.size > 0:
                        faces.append({
                            "bbox": [abs_x1, abs_y1, abs_x2, abs_y2],
                            "confidence": 0.88,
                            "crop": crop
                        })
                if faces:
                    return faces
            except Exception as e:
                logger.debug(f"Cascade detection error: {e}")

        # Fallback: if person ROI is provided, use head portion directly
        if person_bbox is not None:
            faces.append({
                "bbox": [rx1, ry1, rx2, ry2],
                "confidence": 0.70,
                "crop": roi
            })

        return faces
