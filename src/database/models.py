import json
import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from src.config import config

Base = declarative_base()

class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, index=True)
    employee_code = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    department = Column(String(100), default="Operations")
    status = Column(String(20), default="active") # active, inactive
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    face_profiles = relationship("FaceProfile", back_populates="employee", cascade="all, delete-orphan")
    events = relationship("Event", back_populates="employee")

    def to_dict(self):
        return {
            "id": self.id,
            "employee_code": self.employee_code,
            "name": self.name,
            "department": self.department,
            "status": self.status,
            "notes": self.notes,
            "face_profile_count": len(self.face_profiles) if self.face_profiles else 0,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class FaceProfile(Base):
    __tablename__ = "face_profiles"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id", ondelete="CASCADE"), nullable=False)
    sample_index = Column(Integer, default=0) # 0=Front, 1=Left, 2=Right, etc.
    pose_angle = Column(String(20), default="front")
    embedding_json = Column(Text, nullable=False) # JSON array of floats (e.g. 128-d or 512-d)
    model_version = Column(String(50), default="cv_face_embedder_v1")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    employee = relationship("Employee", back_populates="face_profiles")

    @property
    def embedding(self):
        return json.loads(self.embedding_json)

    @embedding.setter
    def embedding(self, val):
        self.embedding_json = json.dumps(val)

class Camera(Base):
    __tablename__ = "cameras"

    id = Column(String(50), primary_key=True) # e.g. 'CAM-01'
    name = Column(String(100), nullable=False)
    source_type = Column(String(20), default="synthetic") # webcam, file, rtsp, synthetic
    source_uri = Column(String(255), default="0")
    location = Column(String(100), default="Main Site")
    fps = Column(Integer, default=15)
    status = Column(String(20), default="active") # active, offline, error
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    events = relationship("Event", back_populates="camera")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "source_type": self.source_type,
            "source_uri": self.source_uri,
            "location": self.location,
            "fps": self.fps,
            "status": self.status,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class Event(Base):
    __tablename__ = "events"

    id = Column(String(50), primary_key=True) # e.g. EVT-20261001-0001
    camera_id = Column(String(50), ForeignKey("cameras.id"), nullable=False)
    track_id = Column(Integer, nullable=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True)
    activity = Column(String(50), nullable=False) # 'falling', 'sitting', 'standing', 'working'
    activity_confidence = Column(Float, default=0.0)
    identity_status = Column(String(20), default="unknown") # 'recognized', 'unknown', 'pending'
    identity_confidence = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String(30), default="pending_review") # 'detected', 'pending_review', 'confirmed', 'rejected'
    snapshot_path = Column(String(255), nullable=True)
    clip_path = Column(String(255), nullable=True)
    notes = Column(Text, nullable=True)

    camera = relationship("Camera", back_populates="events")
    employee = relationship("Employee", back_populates="events")
    reviews = relationship("EventReview", back_populates="event", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "camera_name": self.camera.name if self.camera else self.camera_id,
            "track_id": self.track_id,
            "employee_id": self.employee_id,
            "employee_code": self.employee.employee_code if self.employee else None,
            "employee_name": self.employee.name if self.employee else ("Unknown Person" if self.identity_status == "unknown" else None),
            "activity": self.activity,
            "activity_confidence": round(self.activity_confidence, 2),
            "identity_status": self.identity_status,
            "identity_confidence": round(self.identity_confidence, 2) if self.identity_confidence else 0.0,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S") if self.timestamp else None,
            "status": self.status,
            "snapshot_path": self.snapshot_path,
            "clip_path": self.clip_path,
            "notes": self.notes,
            "review_count": len(self.reviews) if self.reviews else 0
        }

class EventReview(Base):
    __tablename__ = "event_reviews"

    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String(50), ForeignKey("events.id", ondelete="CASCADE"), nullable=False)
    decision = Column(String(30), nullable=False) # 'confirmed', 'rejected'
    note = Column(Text, nullable=True)
    reviewed_by = Column(String(100), default="Supervisor")
    reviewed_at = Column(DateTime, default=datetime.datetime.utcnow)

    event = relationship("Event", back_populates="reviews")

    def to_dict(self):
        return {
            "id": self.id,
            "event_id": self.event_id,
            "decision": self.decision,
            "note": self.note,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None
        }
