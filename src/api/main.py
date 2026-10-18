import os
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Depends, HTTPException, Query, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Body
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
from src.video.camera import verify_camera_source, scan_local_cameras, scan_network_cameras, mask_camera_uri
from src.zones.zone_manager import Zone

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
    
    # Initialize vision pipelines for configured active cameras
    db = SessionLocal()
    try:
        cams = db.query(Camera).filter(Camera.status == "active").all()
        for cam in cams:
            worker = VisionPipelineWorker(
                camera_id=cam.id,
                source_type=cam.source_type,
                source_uri=cam.source_uri,
                fps=cam.fps or 24,
                ai_enabled=cam.ai_enabled if cam.ai_enabled is not None else True
            )
            worker.start()
            pipeline_workers[cam.id] = worker
            logger.info(f"Initialized active pipeline worker for {cam.id} ({cam.source_type})")
    finally:
        db.close()

    yield

    # Graceful Shutdown
    for cam_id, worker in list(pipeline_workers.items()):
        try:
            worker.stop()
        except Exception as e:
            logger.error(f"Error stopping worker {cam_id}: {e}")

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
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
        logger.info("Dashboard WebSocket client disconnected")
    except Exception as e:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
        logger.debug(f"WebSocket closed: {e}")

# ==================== HTML DASHBOARD ROUTES ====================

@app.get("/", response_class=HTMLResponse)
def page_dashboard(request: Request, db: Session = Depends(get_db)):
    cameras = db.query(Camera).all()
    recent_events = db.query(Event).order_by(Event.timestamp.desc()).limit(8).all()
    employees_count = db.query(Employee).filter(Employee.status == "active").count()
    events_count = db.query(Event).count()
    confirmed_falls = db.query(Event).filter(Event.activity == "falling", Event.status == "confirmed").count()

    # Determine real-time camera metadata
    cam_data = []
    online_count = 0
    for c in cameras:
        worker = pipeline_workers.get(c.id)
        d = c.to_dict()
        if worker:
            d["is_online"] = worker.stream.is_online
            d["measured_fps"] = worker.stats.get("fps", worker.fps)
            d["resolution"] = f"{worker.stream.resolution[0]}x{worker.stream.resolution[1]}"
            d["ai_enabled"] = worker.ai_enabled
            d["active_tracks_count"] = worker.stats.get("active_tracks", 0)
            d["tracks"] = worker.current_tracks_info
            if worker.stream.is_online:
                online_count += 1
        else:
            d["is_online"] = False
            d["measured_fps"] = 0
            d["resolution"] = "1280x720"
            d["ai_enabled"] = False
            d["active_tracks_count"] = 0
            d["tracks"] = []
        cam_data.append(d)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "cameras": cam_data,
            "recent_events": [e.to_dict() for e in recent_events],
            "stats": {
                "cameras_online": online_count,
                "total_cameras": len(cameras),
                "employees_enrolled": employees_count,
                "total_events": events_count,
                "confirmed_falls": confirmed_falls
            }
        }
    )

@app.get("/cameras", response_class=HTMLResponse)
def page_cameras(request: Request, db: Session = Depends(get_db)):
    cameras = db.query(Camera).all()
    cam_list = []
    for c in cameras:
        worker = pipeline_workers.get(c.id)
        d = c.to_dict()
        if worker:
            d["is_online"] = worker.stream.is_online
            d["measured_fps"] = worker.stats.get("fps", worker.fps)
            d["resolution"] = f"{worker.stream.resolution[0]}x{worker.stream.resolution[1]}"
            d["ai_enabled"] = worker.ai_enabled
            d["active_tracks_count"] = worker.stats.get("active_tracks", 0)
        else:
            d["is_online"] = False
            d["measured_fps"] = 0
            d["resolution"] = "1280x720"
            d["ai_enabled"] = False
            d["active_tracks_count"] = 0
        cam_list.append(d)

    return templates.TemplateResponse(
        request=request,
        name="cameras.html",
        context={"cameras": cam_list}
    )

@app.get("/people-intelligence", response_class=HTMLResponse)
def page_people_intelligence(request: Request, db: Session = Depends(get_db)):
    cameras = db.query(Camera).all()
    return templates.TemplateResponse(
        request=request,
        name="people_intelligence.html",
        context={"cameras": [c.to_dict() for c in cameras]}
    )

@app.get("/zone-analytics", response_class=HTMLResponse)
def page_zone_analytics(request: Request, db: Session = Depends(get_db)):
    cameras = db.query(Camera).all()
    return templates.TemplateResponse(
        request=request,
        name="zone_analytics.html",
        context={"cameras": [c.to_dict() for c in cameras]}
    )

@app.get("/events", response_class=HTMLResponse)
def page_events(request: Request, db: Session = Depends(get_db)):
    events = db.query(Event).order_by(Event.timestamp.desc()).limit(100).all()
    cameras = db.query(Camera).all()
    return templates.TemplateResponse(
        request=request,
        name="events.html",
        context={
            "events": [e.to_dict() for e in events],
            "cameras": [c.to_dict() for c in cameras]
        }
    )

@app.get("/employees", response_class=HTMLResponse)
def page_employees(request: Request, db: Session = Depends(get_db)):
    employees = db.query(Employee).all()
    return templates.TemplateResponse(
        request=request,
        name="employees.html",
        context={"employees": [emp.to_dict() for emp in employees]}
    )

@app.get("/enrollment", response_class=HTMLResponse)
def page_enrollment(request: Request, db: Session = Depends(get_db)):
    employees = db.query(Employee).all()
    cameras = db.query(Camera).filter(Camera.status == "active").all()
    return templates.TemplateResponse(
        request=request,
        name="enrollment.html",
        context={
            "employees": [emp.to_dict() for emp in employees],
            "cameras": [c.to_dict() for c in cameras]
        }
    )

@app.get("/analytics", response_class=HTMLResponse)
def page_analytics(request: Request):
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

# ==================== CAMERA MANAGEMENT API ====================

@app.get("/api/cameras")
def get_cameras(db: Session = Depends(get_db)):
    cameras = db.query(Camera).all()
    result = []
    for c in cameras:
        d = c.to_dict()
        worker = pipeline_workers.get(c.id)
        if worker:
            d["is_online"] = worker.stream.is_online
            d["fps"] = worker.stats.get("fps", worker.fps)
            d["measured_fps"] = worker.stats.get("fps", worker.fps)
            d["resolution"] = f"{worker.stream.resolution[0]}x{worker.stream.resolution[1]}"
            d["ai_enabled"] = worker.ai_enabled
            d["active_tracks"] = worker.stats.get("active_tracks", 0)
            d["tracks"] = worker.current_tracks_info
        else:
            d["is_online"] = False
            d["fps"] = c.fps or 24
            d["measured_fps"] = 0
            d["resolution"] = "1280x720"
            d["ai_enabled"] = False
            d["active_tracks"] = 0
            d["tracks"] = []
        result.append(d)
    return result

@app.post("/api/cameras/test")
def test_camera_endpoint(payload: Dict[str, Any] = Body(...)):
    """Tests a camera source without adding it to the database."""
    source_type = payload.get("source_type", "webcam")
    source_uri = str(payload.get("source_uri", "0"))
    username = payload.get("username", "")
    password = payload.get("password", "")
    
    result = verify_camera_source(
        source_type=source_type,
        source_uri=source_uri,
        username=username,
        password=password
    )
    return result

@app.post("/api/cameras/scan-local")
def scan_local_devices():
    """Scans local hardware for webcams and USB video capture devices."""
    devices = scan_local_cameras()
    return {"devices": devices, "count": len(devices)}

@app.post("/api/cameras/scan-network")
def scan_network_devices():
    """Scans local network for ONVIF IP camera streams."""
    devices = scan_network_cameras()
    return {"devices": devices, "count": len(devices)}

@app.post("/api/cameras")
def add_camera(payload: Dict[str, Any] = Body(...), db: Session = Depends(get_db)):
    """Registers and starts a new camera stream with optional AI pipeline."""
    cam_id = payload.get("id", "").strip() or f"CAM-{int(time.time()) % 1000:03d}"
    name = payload.get("name", "").strip() or f"Camera {cam_id}"
    source_type = payload.get("source_type", "webcam").lower()
    source_uri = str(payload.get("source_uri", "0")).strip()
    location = payload.get("location", "Main Site").strip()
    fps = int(payload.get("fps", 24))
    ai_enabled = bool(payload.get("ai_enabled", True))
    username = payload.get("username", "")
    password = payload.get("password", "")

    # Embed credentials if RTSP
    if username and password and "rtsp://" in source_uri and "@" not in source_uri:
        source_uri = source_uri.replace("rtsp://", f"rtsp://{username}:{password}@")

    existing = db.query(Camera).filter(Camera.id == cam_id).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Camera with ID '{cam_id}' already exists.")

    cam = Camera(
        id=cam_id,
        name=name,
        source_type=source_type,
        source_uri=source_uri,
        location=location,
        fps=fps,
        ai_enabled=ai_enabled,
        status="active",
        description=payload.get("description", "")
    )
    db.add(cam)
    db.commit()
    db.refresh(cam)

    # Initialize and start pipeline worker
    worker = VisionPipelineWorker(
        camera_id=cam.id,
        source_type=cam.source_type,
        source_uri=cam.source_uri,
        fps=cam.fps or 24,
        ai_enabled=cam.ai_enabled
    )
    worker.start()
    pipeline_workers[cam.id] = worker
    logger.info(f"Added and started camera {cam.id} ({cam.source_type})")

    return {"success": True, "camera": cam.to_dict()}

@app.delete("/api/cameras/{camera_id}")
def delete_camera(camera_id: str, db: Session = Depends(get_db)):
    """Stops worker and deletes camera from database."""
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    worker = pipeline_workers.pop(camera_id, None)
    if worker:
        worker.stop()

    db.delete(cam)
    db.commit()
    return {"success": True, "message": f"Camera {camera_id} deleted."}

@app.post("/api/cameras/{camera_id}/toggle-ai")
def toggle_camera_ai(camera_id: str, db: Session = Depends(get_db)):
    """Toggles AI processing layer on/off for a specific camera."""
    worker = pipeline_workers.get(camera_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Camera worker not running")
    
    new_state = worker.toggle_ai()
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if cam:
        cam.ai_enabled = new_state
        db.commit()

    return {"success": True, "camera_id": camera_id, "ai_enabled": new_state}

@app.post("/api/cameras/{camera_id}/reconnect")
def reconnect_camera(camera_id: str):
    """Forces immediate stream reconnect for a camera."""
    worker = pipeline_workers.get(camera_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Camera worker not running")
    worker.reconnect()
    return {"success": True, "camera_id": camera_id, "is_online": worker.stream.is_online}

@app.get("/api/cameras/{camera_id}/stream")
def stream_camera(camera_id: str, mode: str = Query("ai", pattern="^(ai|raw)$")):
    """Streams MJPEG video feed: ?mode=ai (annotated with YOLO/pose) or ?mode=raw (clean live stream)."""
    worker = pipeline_workers.get(camera_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Camera stream not found or inactive")
    return StreamingResponse(
        worker.get_mjpeg_generator(mode=mode),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/cameras/{camera_id}/status")
def get_camera_status(camera_id: str):
    worker = pipeline_workers.get(camera_id)
    if not worker:
        return {"camera_id": camera_id, "status": "offline", "is_online": False, "ai_enabled": False, "active_tracks": 0, "fps": 0.0}
    return {
        "camera_id": camera_id,
        "status": "active" if worker.stream.is_online else "offline",
        "is_online": worker.stream.is_online,
        "ai_enabled": worker.ai_enabled,
        "fps": worker.stats.get("fps", worker.fps),
        "resolution": f"{worker.stream.resolution[0]}x{worker.stream.resolution[1]}",
        "active_tracks": worker.stats.get("active_tracks", 0),
        "tracks": worker.current_tracks_info
    }

# ==================== PEOPLE INTELLIGENCE API ====================

@app.get("/api/people-intelligence")
def get_people_intelligence(db: Session = Depends(get_db)):
    """Returns real-time telemetry of all tracked subjects across all cameras."""
    subjects = []
    for cam_id, worker in pipeline_workers.items():
        if not worker.stream.is_online:
            continue
        for track in worker.current_tracks_info:
            item = dict(track)
            item["camera_id"] = cam_id
            item["is_demo"] = (worker.source_type == "demo")
            subjects.append(item)
    return {"subjects": subjects, "total_active_people": len(subjects)}

# ==================== MULTI-DOMAIN VISION ANALYTICS API ====================

@app.get("/api/zones")
def get_zones(camera_id: Optional[str] = None):
    """Returns configured zones and live occupancy telemetry across camera streams."""
    results = {}
    for cid, worker in pipeline_workers.items():
        if camera_id and cid != camera_id:
            continue
        results[cid] = worker.zone_manager.get_summary()
    return {"cameras": results}

@app.post("/api/zones")
def add_or_update_zone(payload: Dict[str, Any] = Body(...), db: Session = Depends(get_db)):
    """Dynamically creates or updates an ROI zone on a camera."""
    camera_id = payload.get("camera_id")
    worker = pipeline_workers.get(camera_id)
    if not worker:
        cam = db.query(Camera).filter(Camera.id == camera_id).first()
        if cam:
            worker = VisionPipelineWorker(
                camera_id=cam.id,
                source_type=cam.source_type,
                source_uri=cam.source_uri,
                fps=cam.fps or 24,
                ai_enabled=cam.ai_enabled
            )
            pipeline_workers[cam.id] = worker
        else:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    zone_id = payload.get("id") or f"{camera_id}-Z{int(time.time()) % 1000}"
    name = payload.get("name") or f"Zone {zone_id}"
    zone_type = payload.get("zone_type", "general")
    polygon = payload.get("polygon", [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]])
    capacity = int(payload.get("capacity_limit", 4))
    max_dwell = float(payload.get("max_dwell_sec", 30.0))

    new_zone = Zone(
        zone_id=zone_id,
        camera_id=camera_id,
        name=name,
        zone_type=zone_type,
        polygon=polygon,
        capacity_limit=capacity,
        max_dwell_sec=max_dwell
    )
    worker.zone_manager.add_zone(new_zone)
    return {"success": True, "zone": new_zone.to_dict()}

@app.delete("/api/zones/{camera_id}/{zone_id}")
def delete_zone(camera_id: str, zone_id: str):
    worker = pipeline_workers.get(camera_id)
    if worker:
        worker.zone_manager.remove_zone(zone_id)
    return {"success": True, "message": f"Zone {zone_id} removed"}

@app.post("/api/zones/{camera_id}/toggle-overlay")
def toggle_zone_overlay(camera_id: str):
    worker = pipeline_workers.get(camera_id)
    if not worker:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    worker.draw_zones_enabled = not worker.draw_zones_enabled
    return {"success": True, "draw_zones_enabled": worker.draw_zones_enabled}

@app.get("/api/ppe/status")
def get_ppe_status():
    """Returns workplace safety PPE compliance metrics across all active persons."""
    total_persons = 0
    compliant_count = 0
    violations_list = []

    for cid, worker in pipeline_workers.items():
        if not worker.stream.is_online:
            continue
        for track in worker.current_tracks_info:
            total_persons += 1
            ppe = track.get("ppe", {})
            if ppe.get("is_compliant"):
                compliant_count += 1
            else:
                violations_list.append({
                    "camera_id": cid,
                    "track_id": track.get("track_id"),
                    "employee_code": track.get("employee_code"),
                    "employee_name": track.get("employee_name"),
                    "zone_name": track.get("zone", {}).get("name"),
                    "violations": ppe.get("violations", ["Missing PPE"])
                })

    rate = round((compliant_count / total_persons * 100), 1) if total_persons > 0 else 100.0
    return {
        "total_active_workers": total_persons,
        "compliant_count": compliant_count,
        "violation_count": len(violations_list),
        "compliance_rate_pct": rate,
        "violations": violations_list
    }

@app.get("/api/hospitality/status")
def get_hospitality_status():
    """Returns restaurant/hotel table occupancy, dwell duration, and waiter service visits."""
    all_tables = []
    total_occupied = 0
    total_turnovers = 0
    total_waiter_visits = 0

    for cid, worker in pipeline_workers.items():
        summary = worker.zone_manager.get_summary()
        hosp = summary["hospitality"]
        total_turnovers += hosp["total_turnovers"]
        total_waiter_visits += hosp["total_waiter_visits"]

        for z in summary["zones"]:
            if z["zone_type"] == "table":
                all_tables.append(z)
                if z["occupied"]:
                    total_occupied += 1

    return {
        "total_tables": len(all_tables),
        "occupied_tables": total_occupied,
        "available_tables": len(all_tables) - total_occupied,
        "turnover_count": total_turnovers,
        "total_waiter_visits": total_waiter_visits,
        "tables": all_tables
    }

@app.get("/api/queue/status")
def get_queue_status():
    """Returns retail/healthcare queue headcounts and estimated waiting times."""
    all_queues = []
    total_waiting = 0
    max_wait_time = 0.0

    for cid, worker in pipeline_workers.items():
        summary = worker.zone_manager.get_summary()
        for z in summary["zones"]:
            if z["zone_type"] == "queue":
                all_queues.append(z)
                total_waiting += z["occupant_count"]
                if z["estimated_wait_min"] > max_wait_time:
                    max_wait_time = z["estimated_wait_min"]

    return {
        "total_queues": len(all_queues),
        "total_waiting_people": total_waiting,
        "max_estimated_wait_min": max_wait_time,
        "queues": all_queues
    }

@app.get("/api/parking/status")
def get_parking_status():
    """Returns smart parking space availability and occupancy analytics."""
    all_spaces = []
    total_occupied = 0

    for cid, worker in pipeline_workers.items():
        summary = worker.zone_manager.get_summary()
        for z in summary["zones"]:
            if z["zone_type"] == "parking":
                all_spaces.append(z)
                if z["occupied"]:
                    total_occupied += 1

    rate = round((total_occupied / len(all_spaces) * 100), 1) if all_spaces else 0.0
    return {
        "total_spaces": len(all_spaces),
        "occupied_spaces": total_occupied,
        "available_spaces": len(all_spaces) - total_occupied,
        "occupancy_rate_pct": rate,
        "spaces": all_spaces
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

@app.post("/api/events/{event_id}/rejected")
def reject_event(event_id: str, note: Optional[str] = None, reviewer: str = "Supervisor", db: Session = Depends(get_db)):
    event = event_manager.review_event(db, event_id, decision="rejected", note=note, reviewed_by=reviewer)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return {"success": True, "event": event.to_dict()}

@app.post("/api/events/simulate")
def simulate_fall_event(camera_id: str = "CAM-01", db: Session = Depends(get_db)):
    """Dispatches a test fall event for integration verification."""
    worker = pipeline_workers.get(camera_id)
    raw_frame = worker.get_latest_frame() if worker else None
    buffer = worker.video_buffer if worker else None
    
    event = event_manager.create_event(
        db=db,
        camera_id=camera_id,
        track_id=99,
        employee_id=1,
        activity="falling",
        activity_confidence=0.96,
        identity_confidence=0.88,
        identity_status="recognized",
        current_frame=raw_frame,
        video_buffer=buffer,
        notes="Automated safety evaluation fall simulation test."
    )
    return {"success": True, "event": event.to_dict()}

# ==================== EMPLOYEE & BIOMETRIC ENROLLMENT API ====================

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
        raise HTTPException(status_code=400, detail="Employee code already registered")
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
    return emp.to_dict()

@app.post("/api/enrollment/upload")
async def enroll_face_sample(
    employee_id: int = Form(...),
    sample_index: int = Form(0),
    pose_angle: str = Form("front"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    emp = db.query(Employee).filter(Employee.id == employee_id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    contents = await file.read()
    result = enrollment_service.process_and_enroll_image_bytes(
        image_bytes=contents,
        employee_id=employee_id,
        db=db,
        sample_index=sample_index,
        pose_angle=pose_angle
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Enrollment failed"))
    
    # Trigger face recognizer sync in all workers
    for worker in pipeline_workers.values():
        worker.face_recognizer.sync_gallery(db)

    return result

@app.get("/api/system/status")
def get_system_status(db: Session = Depends(get_db)):
    """Returns platform health, inference engines, and camera telemetry."""
    active_cams = sum(1 for w in pipeline_workers.values() if w.stream.is_online)
    first_worker = next(iter(pipeline_workers.values()), None)
    pose_engine = first_worker.pose_estimator.backend if first_worker else "yolo_pose"
    classifier_loaded = first_worker.activity_classifier.is_loaded if first_worker else False

    return {
        "status": "operational",
        "cameras_online": active_cams,
        "total_cameras": len(pipeline_workers),
        "ai_workers_active": sum(1 for w in pipeline_workers.values() if w.ai_enabled),
        "pose_engine": pose_engine,
        "classifier_loaded": classifier_loaded,
        "classifier_type": "Random Forest v1.0",
        "embedder_type": "TorchVision Deep MobileNetV2 (128-d)",
        "websocket_clients": len(active_websockets)
    }
