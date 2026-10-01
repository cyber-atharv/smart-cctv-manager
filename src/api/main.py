import os
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Depends, HTTPException, Query, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from sqlalchemy.orm import Session
import logging

from src.config import config
from src.database.db import get_db, init_db, SessionLocal
from src.database.models import Camera, Employee, FaceProfile, Event, EventReview
from src.events.event_manager import event_manager
from src.face.enrollment import FaceEnrollmentService
from src.video.pipeline import VisionPipelineWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("cctv.server")

from contextlib import asynccontextmanager

# Active Vision Pipeline Workers: camera_id -> VisionPipelineWorker
pipeline_workers: Dict[str, VisionPipelineWorker] = {}
enrollment_service = FaceEnrollmentService()

# WebSocket active client pool
active_websockets: List[WebSocket] = []
loop: Optional[asyncio.AbstractEventLoop] = None

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    init_db()
    
    # Initialize vision pipelines for configured cameras
    db = SessionLocal()
    try:
        cams = db.query(Camera).filter(Camera.status == "active").all()
        for cam in cams:
            worker = VisionPipelineWorker(
                camera_id=cam.id,
                source_type=cam.source_type,
                source_uri=cam.source_uri,
                fps=cam.fps or 15
            )
            worker.start()
            pipeline_workers[cam.id] = worker
            logger.info(f"Initialized active pipeline worker for {cam.id}")
    finally:
        db.close()

    yield

    # Graceful Shutdown
    for cam_id, worker in pipeline_workers.items():
        worker.stop()

app = FastAPI(
    title=config.system.get("app", {}).get("name", "Smart CCTV AI Safety Platform"),
    version=config.system.get("app", {}).get("version", "1.0.0"),
    description="Smart CCTV AI Safety & Identification Platform (Fall Detection + Activity Recognition + Tracking + Face Recognition)",
    lifespan=lifespan
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
STORAGE_DIR = BASE_DIR / "storage"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/storage", StaticFiles(directory=str(STORAGE_DIR)), name="storage")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

def broadcast_alert(event_data: Dict[str, Any]):
    """Dispatches real-time incident event to all connected dashboard WebSockets."""
    dead_sockets = []
    for ws in active_websockets:
        try:
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    ws.send_json({"type": "SAFETY_ALERT", "data": event_data}),
                    loop
                )
        except Exception:
            dead_sockets.append(ws)
    for ws in dead_sockets:
        if ws in active_websockets:
            active_websockets.remove(ws)

event_manager.register_listener(broadcast_alert)

# ==================== WEBSOCKET ENDPOINT ====================

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    logger.info(f"Dashboard WebSocket client connected (Total: {len(active_websockets)})")
    try:
        while True:
            # Keep-alive ping/pong
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
        logger.info("Dashboard WebSocket client disconnected")

# ==================== HTML PAGE ROUTES ====================

@app.get("/", response_class=HTMLResponse)
def page_dashboard(request: Request, db: Session = Depends(get_db)):
    cameras = db.query(Camera).all()
    recent_events = db.query(Event).order_by(Event.timestamp.desc()).limit(10).all()
    employees_count = db.query(Employee).count()
    events_count = db.query(Event).count()
    confirmed_falls = db.query(Event).filter(Event.status == "confirmed").count()
    
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "cameras": [c.to_dict() for c in cameras],
            "recent_events": [e.to_dict() for e in recent_events],
            "stats": {
                "cameras_online": len([c for c in cameras if c.status == 'active']),
                "employees_enrolled": employees_count,
                "total_events": events_count,
                "confirmed_falls": confirmed_falls
            }
        }
    )

@app.get("/cameras", response_class=HTMLResponse)
def page_cameras(request: Request, db: Session = Depends(get_db)):
    cameras = db.query(Camera).all()
    return templates.TemplateResponse(
        request=request,
        name="cameras.html",
        context={"cameras": [c.to_dict() for c in cameras]}
    )

@app.get("/events", response_class=HTMLResponse)
def page_events(request: Request, db: Session = Depends(get_db)):
    events = db.query(Event).order_by(Event.timestamp.desc()).all()
    return templates.TemplateResponse(
        request=request,
        name="events.html",
        context={"events": [e.to_dict() for e in events]}
    )

@app.get("/employees", response_class=HTMLResponse)
def page_employees(request: Request, db: Session = Depends(get_db)):
    employees = db.query(Employee).all()
    return templates.TemplateResponse(
        request=request,
        name="employees.html",
        context={"employees": [e.to_dict() for e in employees]}
    )

@app.get("/enrollment", response_class=HTMLResponse)
def page_enrollment(request: Request, db: Session = Depends(get_db)):
    employees = db.query(Employee).all()
    return templates.TemplateResponse(
        request=request,
        name="enrollment.html",
        context={"employees": [e.to_dict() for e in employees]}
    )

@app.get("/analytics", response_class=HTMLResponse)
def page_analytics(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse(
        request=request,
        name="analytics.html",
        context={}
    )

@app.get("/settings", response_class=HTMLResponse)
def page_settings(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="settings.html",
        context={
            "system_config": config.system,
            "thresholds_config": config.thresholds
        }
    )

# ==================== CAMERA API ====================

@app.get("/api/cameras")
def get_cameras(db: Session = Depends(get_db)):
    cameras = db.query(Camera).all()
    result = []
    for c in cameras:
        d = c.to_dict()
        worker = pipeline_workers.get(c.id)
        if worker:
            d["fps"] = worker.stats.get("fps", 15.0)
            d["active_tracks"] = worker.stats.get("active_tracks", 0)
            d["tracks"] = worker.current_tracks_info
        else:
            d["active_tracks"] = 0
            d["tracks"] = []
        result.append(d)
    return result

@app.get("/api/cameras/{camera_id}/stream")
def stream_camera(camera_id: str):
    worker = pipeline_workers.get(camera_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Camera stream not found or inactive")
    return StreamingResponse(
        worker.get_mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/cameras/{camera_id}/status")
def get_camera_status(camera_id: str):
    worker = pipeline_workers.get(camera_id)
    if not worker:
        return {"camera_id": camera_id, "status": "offline", "active_tracks": 0, "fps": 0.0}
    return {
        "camera_id": camera_id,
        "status": "active",
        "fps": worker.stats.get("fps", 15.0),
        "active_tracks": worker.stats.get("active_tracks", 0),
        "tracks": worker.current_tracks_info
    }

# ==================== SAFETY EVENTS API ====================

@app.get("/api/events")
def get_events(
    status: Optional[str] = None,
    camera_id: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    query = db.query(Event)
    if status:
        query = query.filter(Event.status == status)
    if camera_id:
        query = query.filter(Event.camera_id == camera_id)
    events = query.order_by(Event.timestamp.desc()).limit(limit).all()
    return [e.to_dict() for e in events]

@app.get("/api/events/{event_id}")
def get_event_detail(event_id: str, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    data = event.to_dict()
    data["reviews"] = [r.to_dict() for r in event.reviews]
    return data

@app.post("/api/events/{event_id}/confirm")
def confirm_event(event_id: str, note: Optional[str] = None, reviewer: str = "Supervisor", db: Session = Depends(get_db)):
    event = event_manager.review_event(db, event_id, decision="confirmed", note=note, reviewed_by=reviewer)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return {"success": True, "event": event.to_dict()}

@app.post("/api/events/{event_id}/reject")
def reject_event(event_id: str, note: Optional[str] = None, reviewer: str = "Supervisor", db: Session = Depends(get_db)):
    event = event_manager.review_event(db, event_id, decision="rejected", note=note, reviewed_by=reviewer)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return {"success": True, "event": event.to_dict()}

@app.post("/api/events/simulate")
def simulate_fall_event(
    camera_id: str = "CAM-01",
    employee_code: Optional[str] = "EMP001",
    db: Session = Depends(get_db)
):
    """Triggers an instantaneous safety fall incident event for test/demonstration purposes."""
    emp = db.query(Employee).filter(Employee.employee_code == employee_code).first() if employee_code else None
    
    worker = pipeline_workers.get(camera_id)
    raw_frame = worker.get_latest_frame() if worker else None
    buffer = worker.video_buffer if worker else None

    evt = event_manager.create_event(
        db=db,
        camera_id=camera_id,
        track_id=7,
        employee_id=emp.id if emp else None,
        activity="falling",
        activity_confidence=0.96,
        identity_confidence=0.92 if emp else 0.0,
        identity_status="recognized" if emp else "unknown",
        current_frame=raw_frame,
        video_buffer=buffer,
        notes="Simulated Fall Event (Demonstration Scenario)"
    )
    return {"success": True, "event": evt.to_dict()}

# ==================== EMPLOYEES & BIOMETRICS API ====================

@app.get("/api/employees")
def get_employees(db: Session = Depends(get_db)):
    employees = db.query(Employee).all()
    return [e.to_dict() for e in employees]

@app.post("/api/employees")
def create_employee(
    employee_code: str = Form(...),
    name: str = Form(...),
    department: str = Form("Operations"),
    notes: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    existing = db.query(Employee).filter(Employee.employee_code == employee_code).first()
    if existing:
        raise HTTPException(status_code=400, detail="Employee code already exists")

    emp = Employee(
        employee_code=employee_code,
        name=name,
        department=department,
        status="active",
        notes=notes
    )
    db.add(emp)
    db.commit()
    db.refresh(emp)
    return {"success": True, "employee": emp.to_dict()}

@app.post("/api/face/enroll")
async def enroll_face(
    employee_id: int = Form(...),
    pose_angle: str = Form("front"),
    sample_index: int = Form(0),
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    contents = await image.read()
    res = enrollment_service.enroll_employee_sample(
        db=db,
        employee_id=employee_id,
        image_bytes_or_mat=contents,
        pose_angle=pose_angle,
        sample_index=sample_index
    )
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error", "Enrollment failed"))
    
    # Refresh gallery in all active workers
    for worker in pipeline_workers.values():
        worker.face_recognizer.sync_gallery(db)

    return res

@app.get("/api/face/status")
def get_face_gallery_status(db: Session = Depends(get_db)):
    total_profiles = db.query(FaceProfile).count()
    enrolled_employees = db.query(Employee).join(FaceProfile).distinct().count()
    total_employees = db.query(Employee).count()
    
    return {
        "total_face_profiles": total_profiles,
        "enrolled_employees": enrolled_employees,
        "total_employees": total_employees,
        "model_version": "cv_face_embedder_v1",
        "embedding_dim": 128
    }

# ==================== SYSTEM & ANALYTICS API ====================

@app.get("/api/system/status")
def get_system_status():
    import psutil
    metadata_path = config.models_dir / "metadata.json"
    meta = {}
    if metadata_path.exists():
        import json
        with open(metadata_path, "r") as f:
            meta = json.load(f)

    return {
        "status": "healthy",
        "active_cameras": len(pipeline_workers),
        "total_active_tracks": sum(w.stats.get("active_tracks", 0) for w in pipeline_workers.values()),
        "cpu_usage_pct": psutil.cpu_percent(interval=None),
        "memory_usage_pct": psutil.virtual_memory().percent,
        "ml_model": meta.get("model_name", "RandomForestClassifier"),
        "ml_accuracy": meta.get("metrics", {}).get("test_accuracy", 1.0),
        "version": config.system.get("app", {}).get("version", "1.0.0")
    }

@app.get("/api/analytics/summary")
def get_analytics_summary(db: Session = Depends(get_db)):
    events = db.query(Event).all()
    activity_counts = {"standing": 0, "sitting": 0, "working": 0, "falling": 0}
    status_counts = {"pending_review": 0, "confirmed": 0, "rejected": 0}

    for e in events:
        act = e.activity.lower()
        if act in activity_counts:
            activity_counts[act] += 1
        st = e.status.lower()
        if st in status_counts:
            status_counts[st] += 1

    return {
        "total_events": len(events),
        "activity_distribution": activity_counts,
        "status_distribution": status_counts,
        "active_workers": len(pipeline_workers)
    }
