import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger("cctv.detection")

class PersonDetector:
    """
    Modular Person Detector supporting Ultralytics YOLOv8 with graceful fallback to OpenCV HOG / DNN.
    Filters and returns bounding boxes, confidences, and class IDs exclusively for 'person'.
    """
    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.45):
        self.conf_threshold = conf_threshold
        self.model_name = model_name
        self.backend = "yolo"
        self.yolo_model = None
        self.hog = None

        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(model_name)
            logger.info(f"Loaded YOLOv8 model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load YOLOv8 model ({e}). Initializing OpenCV detector fallback.")
            self.backend = "hog"
            if hasattr(cv2, "HOGDescriptor"):
                try:
                    self.hog = cv2.HOGDescriptor()
                    self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                except Exception as ex:
                    logger.warning(f"Failed to initialize HOGDescriptor: {ex}")
                    self.hog = None
            else:
                self.hog = None

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects persons in the given BGR frame.
        Filters out low-confidence blips, window slats, and implausible non-human contours.
        Returns list of detections: [{'bbox': [x1, y1, x2, y2], 'confidence': float, 'class_name': 'person'}]
        """
        detections = []
        if frame is None or frame.size == 0:
            return detections

        h, w = frame.shape[:2]
        min_area = int(0.012 * w * h) # At least 1.2% of frame area

        if self.backend == "yolo" and self.yolo_model is not None:
            try:
                results = self.yolo_model(frame, imgsz=320, verbose=False, conf=self.conf_threshold)
                for res in results:
                    boxes = res.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        # Class 0 in COCO is 'person'
                        if cls_id == 0:
                            conf = float(box.conf[0].item())
                            xyxy = box.xyxy[0].cpu().numpy().astype(int)
                            x1, y1, x2, y2 = max(0, xyxy[0]), max(0, xyxy[1]), min(w, xyxy[2]), min(h, xyxy[3])
                            bw, bh = x2 - x1, y2 - y1
                            # Filter out tiny artifacts or bizarre aspect ratios
                            if bw >= 28 and bh >= 45 and (bw * bh) >= min_area:
                                ar = bh / max(1.0, float(bw))
                                if 0.45 <= ar <= 4.8:
                                    detections.append({
                                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                        "confidence": round(conf, 3),
                                        "class_name": "person"
                                    })
                return detections
            except Exception as e:
                logger.error(f"YOLO inference error: {e}, falling back to HOG")
                if self.hog is None and hasattr(cv2, "HOGDescriptor"):
                    try:
                        self.hog = cv2.HOGDescriptor()
                        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                    except Exception:
                        self.hog = None

        # Fallback OpenCV HOG detection with strict human-geometry checks
        if self.hog is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes, weights = self.hog.detectMultiScale(gray, winStride=(8, 8), padding=(4, 4), scale=1.05)
            for (x, y, bw, bh), conf in zip(boxes, weights):
                confidence = float(conf) if isinstance(conf, (int, float)) else float(conf[0])
                if confidence >= max(0.65, self.conf_threshold):
                    ar = bh / max(1.0, float(bw))
                    if 1.15 <= ar <= 3.8 and (bw * bh) >= min_area:
                        detections.append({
                            "bbox": [int(x), int(y), int(min(w, x + bw)), int(min(h, y + bh))],
                            "confidence": round(min(1.0, max(0.4, confidence)), 3),
                            "class_name": "person"
                        })

        return detections
