from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from src.config import config
from src.database.models import Base, Camera, Employee, FaceProfile, Event
import datetime

# Ensure connect_args for SQLite
connect_args = {"check_same_thread": False} if "sqlite" in config.database_url else {}
engine = create_engine(config.database_url, echo=False, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initializes schema and seeds default camera and employee records if empty."""
    Base.metadata.create_all(bind=engine)
    
    db: Session = SessionLocal()
    try:
        # Seed Cameras from cameras.yaml
        for cam_cfg in config.cameras:
            existing = db.query(Camera).filter(Camera.id == cam_cfg["id"]).first()
            if not existing:
                cam = Camera(
                    id=cam_cfg["id"],
                    name=cam_cfg.get("name", cam_cfg["id"]),
                    source_type=cam_cfg.get("source_type", "synthetic"),
                    source_uri=str(cam_cfg.get("source_uri", "0")),
                    location=cam_cfg.get("location", "Facility"),
                    fps=cam_cfg.get("fps", 15),
                    status="active",
                    description=cam_cfg.get("description", "")
                )
                db.add(cam)

        # Seed Sample Employees if none exist
        if db.query(Employee).count() == 0:
            emp1 = Employee(
                employee_code="EMP001",
                name="Rahul Sharma",
                department="Operations & Safety",
                status="active",
                notes="Senior Safety Officer - Zone North"
            )
            emp2 = Employee(
                employee_code="EMP002",
                name="Ananya Verma",
                department="Logistics & Warehousing",
                status="active",
                notes="Floor Supervisor"
            )
            emp3 = Employee(
                employee_code="EMP003",
                name="David Miller",
                department="Maintenance",
                status="active",
                notes="Facility Technician"
            )
            db.add_all([emp1, emp2, emp3])
            db.flush()

        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error initializing DB: {e}")
    finally:
        db.close()
