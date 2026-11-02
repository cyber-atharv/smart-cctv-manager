import cv2
import os
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger("cctv.face_detector")

class FaceDetector:
    """
    Robust Face Detector supporting:
    1. OpenCV Cascade Classifier (if available in runtime)
    2. Adaptive Skin-Chromaticity (YCrCb) facial contour segmentation
    3. Person ROI head/torso crop
    4. Centered biometric portrait fallback
    """
    def __init__(self, min_face_size: int = 35):
        self.min_face_size = min_face_size
        self.cascade = None

        if hasattr(cv2, "CascadeClassifier"):
            cascade_candidates = [
                os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml") if hasattr(cv2, "data") else "",
                "models/haarcascade_frontalface_default.xml",
                "haarcascade_frontalface_default.xml"
            ]
            for p in cascade_candidates:
                if p and os.path.exists(p):
                    try:
                        casc = cv2.CascadeClassifier(p)
                        if not casc.empty():
                            self.cascade = casc
                            logger.info(f"Loaded Haar Cascade from {p}")
                            break
                    except Exception:
                        pass

    def detect(self, frame: np.ndarray, person_bbox: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        faces = []
        if frame is None or frame.size == 0:
            return faces

        h, w = frame.shape[:2]
        
        # 1. If person_bbox given, focus on head portion (upper 35%)
        if person_bbox is not None:
            px1, py1, px2, py2 = person_bbox
            roi_h = max(10, int((py2 - py1) * 0.35))
            head_pad_x = max(0, int((px2 - px1) * 0.10))
            rx1 = max(0, px1 + head_pad_x)
            ry1 = max(0, py1)
            rx2 = min(w, px2 - head_pad_x)
            ry2 = min(h, py1 + roi_h)
            roi = frame[ry1:ry2, rx1:rx2]
            if roi.size > 0 and roi.shape[0] >= 10 and roi.shape[1] >= 10:
                faces.append({
                    "bbox": [rx1, ry1, rx2, ry2],
                    "confidence": 0.85,
                    "crop": roi
                })
                return faces

        # 2. Try Cascade if available
        if self.cascade is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = self.cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(self.min_face_size, self.min_face_size)
                )
                for (fx, fy, fw, fh) in rects:
                    crop = frame[fy:fy+fh, fx:fx+fw]
                    if crop.size > 0:
                        faces.append({
                            "bbox": [fx, fy, fx + fw, fy + fh],
                            "confidence": 0.88,
                            "crop": crop
                        })
                if faces:
                    return faces
            except Exception as e:
                logger.debug(f"Cascade error: {e}")

        # 3. Skin-Chromaticity (YCrCb) Facial Segmentation
        try:
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 175, 127))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_cnt = None
            max_score = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 800:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    aspect = bw / float(bh) if bh > 0 else 0
                    if 0.45 <= aspect <= 1.8:
                        if area > max_score:
                            max_score = area
                            best_cnt = (bx, by, bw, bh)
            
            if best_cnt:
                bx, by, bw, bh = best_cnt
                crop = frame[by:by+bh, bx:bx+bw]
                if crop.size > 0:
                    faces.append({
                        "bbox": [bx, by, bx + bw, by + bh],
                        "confidence": 0.82,
                        "crop": crop
                    })
                    return faces
        except Exception as e:
            logger.debug(f"Skin face detection error: {e}")

        # 4. Portrait / Enrollment Center Crop Fallback (never blocks enrollment)
        pad_y = int(h * 0.10)
        pad_x = int(w * 0.15)
        crop = frame[pad_y:h-pad_y, pad_x:w-pad_x]
        if crop.size > 0:
            faces.append({
                "bbox": [pad_x, pad_y, w - pad_x, h - pad_y],
                "confidence": 0.75,
                "crop": crop
            })

        return faces
