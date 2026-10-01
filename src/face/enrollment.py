import cv2
import numpy as np
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
import logging
from src.database.models import Employee, FaceProfile
from src.face.detector import FaceDetector
from src.face.embedder import FaceEmbedder

logger = logging.getLogger("cctv.enrollment")

class FaceEnrollmentService:
    """
    Employee Face Registration and Profile Enrollment Service.
    Enforces biometric validation, multi-angle sample capture, quality metrics,
    and stores L2-normalized embeddings without needlessly retaining raw biometric image files.
    """
    def __init__(self):
        self.detector = FaceDetector(min_face_size=50)
        self.embedder = FaceEmbedder()

    def enroll_employee_sample(
        self,
        db: Session,
        employee_id: int,
        image_bytes_or_mat: Any,
        pose_angle: str = "front",
        sample_index: int = 0
    ) -> Dict[str, Any]:
        """
        Processes a single face image, verifies quality, extracts embedding, and saves to database.
        """
        emp = db.query(Employee).filter(Employee.id == employee_id).first()
        if not emp:
            return {"success": False, "error": f"Employee ID {employee_id} not found."}

        # Convert image input
        if isinstance(image_bytes_or_mat, bytes):
            nparr = np.frombuffer(image_bytes_or_mat, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_bytes_or_mat, np.ndarray):
            frame = image_bytes_or_mat
        else:
            return {"success": False, "error": "Invalid image format provided."}

        if frame is None or frame.size == 0:
            return {"success": False, "error": "Could not decode image."}

        # Detect face
        faces = self.detector.detect(frame)
        if len(faces) == 0:
            return {"success": False, "error": "No face detected in image. Please center face in camera."}
        if len(faces) > 1:
            return {"success": False, "error": "Multiple faces detected. Please ensure only 1 person is in frame."}

        face_data = faces[0]
        crop = face_data["crop"]

        # Quality checks: sharpness and size
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 30.0:
            return {"success": False, "error": "Face image is too blurry. Please hold steady and improve lighting."}

        # Extract embedding
        embedding = self.embedder.get_embedding(crop)
        if not embedding:
            return {"success": False, "error": "Failed to compute biometric embedding."}

        # Delete existing profile for same sample index if re-enrolling
        existing = db.query(FaceProfile).filter(
            FaceProfile.employee_id == employee_id,
            FaceProfile.sample_index == sample_index
        ).first()

        if existing:
            existing.embedding = embedding
            existing.pose_angle = pose_angle
            profile = existing
        else:
            profile = FaceProfile(
                employee_id=employee_id,
                sample_index=sample_index,
                pose_angle=pose_angle,
                model_version=self.embedder.model_version
            )
            profile.embedding = embedding
            db.add(profile)

        db.commit()
        db.refresh(profile)

        total_samples = db.query(FaceProfile).filter(FaceProfile.employee_id == employee_id).count()

        logger.info(f"Enrolled sample #{sample_index} ({pose_angle}) for Employee {emp.employee_code} ({emp.name})")
        return {
            "success": True,
            "profile_id": profile.id,
            "employee_id": employee_id,
            "employee_code": emp.employee_code,
            "sample_index": sample_index,
            "pose_angle": pose_angle,
            "total_samples": total_samples,
            "message": f"Successfully enrolled {pose_angle} sample for {emp.name}"
        }
