import datetime
import time
import asyncio
from typing import Dict, Any, List, Optional, Callable
from sqlalchemy.orm import Session
import logging
from src.database.models import Event, EventReview, Camera, Employee
from src.events.clip_manager import ClipManager, RollingVideoBuffer

logger = logging.getLogger("cctv.event_manager")

class EventManager:
    """
    Central Safety Event Correlation Engine.
    Constructs unified incident records correlating:
    Camera + Track ID + Employee Identity (or Unknown) + Posture/Activity + Confidence + Temporal Verification Evidence
    """
    def __init__(self):
        self.clip_manager = ClipManager()
        self.listeners: List[Callable[[Dict[str, Any]], None]] = []
        self._counter = 1

    def register_listener(self, callback: Callable[[Dict[str, Any]], None]):
        """Registers a listener for real-time WebSocket / SSE alerts."""
        self.listeners.append(callback)

    def _generate_event_id(self) -> str:
        date_str = datetime.datetime.utcnow().strftime("%Y%m%d")
        timestamp_ms = int(time.time() * 1000) % 10000
        event_id = f"EVT-{date_str}-{timestamp_ms:04d}"
        return event_id

    def create_event(
        self,
        db: Session,
        camera_id: str,
        track_id: Optional[int],
        employee_id: Optional[int],
        activity: str,
        activity_confidence: float,
        identity_confidence: float,
        identity_status: str,
        current_frame: Optional[Any] = None,
        video_buffer: Optional[RollingVideoBuffer] = None,
        notes: Optional[str] = None
    ) -> Event:
        """
        Creates and persists a structured Safety Event, captures snapshot and clip,
        and notifies live dashboard subscribers.
        """
        event_id = self._generate_event_id()
        now = datetime.datetime.utcnow()

        # Capture evidence snapshot and video clip
        snapshot_rel_path = ""
        clip_rel_path = ""

        if current_frame is not None:
            snapshot_rel_path = self.clip_manager.save_snapshot(current_frame, event_id)

        if video_buffer is not None:
            clip_rel_path = self.clip_manager.save_clip_async(
                event_id=event_id,
                buffer=video_buffer,
                trigger_time=time.time(),
                pre_seconds=5,
                post_seconds=7,
                fps=15
            )

        event = Event(
            id=event_id,
            camera_id=camera_id,
            track_id=track_id,
            employee_id=employee_id,
            activity=activity,
            activity_confidence=activity_confidence,
            identity_confidence=identity_confidence,
            identity_status=identity_status,
            timestamp=now,
            status="pending_review",
            snapshot_path=snapshot_rel_path,
            clip_path=clip_rel_path,
            notes=notes
        )

        db.add(event)
        db.commit()
        db.refresh(event)

        logger.warning(
            f"SAFETY EVENT CREATED: {event_id} | Camera: {camera_id} | Track: {track_id} | "
            f"Employee: {employee_id or 'Unknown'} | Activity: {activity} ({activity_confidence:.2f})"
        )

        # Notify active listeners
        event_data = event.to_dict()
        for listener in self.listeners:
            try:
                listener(event_data)
            except Exception as e:
                logger.error(f"Error notifying listener: {e}")

        return event

    def review_event(
        self,
        db: Session,
        event_id: str,
        decision: str, # 'confirmed' or 'rejected'
        note: Optional[str] = None,
        reviewed_by: str = "Supervisor"
    ) -> Optional[Event]:
        """Processes supervisor review action (confirm/reject) and updates event state."""
        event = db.query(Event).filter(Event.id == event_id).first()
        if not event:
            return None

        event.status = decision
        if note:
            event.notes = (event.notes + f" | Review: {note}") if event.notes else f"Review: {note}"

        review_entry = EventReview(
            event_id=event_id,
            decision=decision,
            note=note,
            reviewed_by=reviewed_by,
            reviewed_at=datetime.datetime.utcnow()
        )
        db.add(review_entry)
        db.commit()
        db.refresh(event)

        logger.info(f"Event {event_id} reviewed: {decision} by {reviewed_by}")
        return event

event_manager = EventManager()
