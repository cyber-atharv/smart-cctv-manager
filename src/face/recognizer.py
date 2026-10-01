import cv2
import numpy as np
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional, Tuple
import logging
from src.database.models import Employee, FaceProfile
from src.face.detector import FaceDetector
from src.face.embedder import FaceEmbedder

logger = logging.getLogger("cctv.recognizer")

class FaceRecognizer:
    """
    Live Face Recognition Subsystem.
    Compares live CCTV face embeddings against enrolled Employee Gallery.
    Returns employee identity when confidence threshold is satisfied; otherwise returns Unknown.
    """
    def __init__(self, match_threshold: float = 0.58):
        self.match_threshold = match_threshold
        self.detector = FaceDetector(min_face_size=35)
        self.embedder = FaceEmbedder()
        self.gallery: List[Dict[str, Any]] = [] # [{'employee_id': int, 'code': str, 'name': str, 'embedding': list}]
        self.last_sync_time = 0.0

    def sync_gallery(self, db: Session):
        """Loads and caches all enrolled face profiles from database into memory."""
        try:
            profiles = db.query(FaceProfile).all()
            new_gallery = []
            for p in profiles:
                if p.employee and p.employee.status == "active":
                    new_gallery.append({
                        "employee_id": p.employee.id,
                        "employee_code": p.employee.employee_code,
                        "employee_name": p.employee.name,
                        "embedding": p.embedding,
                        "sample_index": p.sample_index
                    })
            self.gallery = new_gallery
            logger.info(f"Synchronized Face Gallery: {len(self.gallery)} active biometric profiles cached.")
        except Exception as e:
            logger.error(f"Failed to sync face gallery: {e}")

    def recognize_in_person_roi(
        self,
        frame: np.ndarray,
        person_bbox: List[int]
    ) -> Tuple[Optional[int], Optional[str], str, float]:
        """
        Detects face inside upper person bounding box and identifies person against employee gallery.
        Returns: (employee_id, employee_code, employee_name, similarity_confidence)
        If not recognized, returns (None, None, "Unknown", 0.0)
        """
        if not self.gallery or frame is None or frame.size == 0:
            return None, None, "Unknown", 0.0

        faces = self.detector.detect(frame, person_bbox=person_bbox)
        if not faces:
            return None, None, "Unknown", 0.0

        face_data = faces[0]
        live_embedding = self.embedder.get_embedding(face_data["crop"])
        if not live_embedding:
            return None, None, "Unknown", 0.0

        # Cosine similarity against all enrolled profiles in gallery
        best_score = -1.0
        best_match = None

        for item in self.gallery:
            sim = FaceEmbedder.compute_similarity(live_embedding, item["embedding"])
            if sim > best_score:
                best_score = sim
                best_match = item

        if best_match and best_score >= self.match_threshold:
            return (
                best_match["employee_id"],
                best_match["employee_code"],
                best_match["employee_name"],
                round(best_score, 3)
            )

        return None, None, "Unknown", round(max(0.0, best_score), 3)
