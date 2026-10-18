import cv2
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger("cctv.zones")

class Zone:
    """
    Polygonal Region of Interest (ROI) with multi-domain monitoring capabilities:
    - 'table': Hospitality dining table occupancy, duration, and waiter service
    - 'queue': Retail/hospital queue line count and waiting-time estimation
    - 'parking': Smart parking space availability detection
    - 'restricted': Security perimeter intrusion monitoring
    - 'loitering': Security anomaly dwell-time tracking
    """
    def __init__(
        self,
        zone_id: str,
        camera_id: str,
        name: str,
        zone_type: str = "general",
        polygon: Optional[List[List[float]]] = None,
        capacity_limit: int = 4,
        max_dwell_sec: float = 30.0,
        enabled: bool = True
    ):
        self.id = zone_id
        self.camera_id = camera_id
        self.name = name
        self.zone_type = zone_type.lower()
        # Normalized coordinates [ [x1, y1], [x2, y2], ... ] (0.0 to 1.0)
        self.polygon = polygon or [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]]
        self.capacity_limit = capacity_limit
        self.max_dwell_sec = max_dwell_sec
        self.enabled = enabled

        # Real-time state
        self.occupied = False
        self.occupant_count = 0
        self.occupant_track_ids: List[int] = []
        self.first_occupied_time: Optional[float] = None
        self.current_dwell_sec: float = 0.0
        
        # Turnovers and history
        self.total_visits = 0
        self.completed_dwell_durations: List[float] = []
        self.waiter_visits = 0
        self.last_waiter_visit_time: Optional[float] = None
        
        # Alerts
        self.is_alert = False
        self.alert_message = ""
        self.last_alert_time: float = 0.0

    def update(self, tracks: List[Any], frame_shape: Tuple[int, int, int]):
        if not self.enabled:
            return

        h, w = frame_shape[:2]
        pts_pixel = np.array([[int(p[0] * w), int(p[1] * h)] for p in self.polygon], dtype=np.int32)

        current_occupants = []
        has_staff_visit = False

        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            # Test person feet / ground position (most accurate for floor zones)
            feet_pt = ((x1 + x2) / 2.0, float(y2))
            center_pt = (track.center_x, track.center_y)

            # Test if inside polygon
            dist_feet = cv2.pointPolygonTest(pts_pixel, (int(feet_pt[0]), int(feet_pt[1])), False)
            dist_center = cv2.pointPolygonTest(pts_pixel, (int(center_pt[0]), int(center_pt[1])), False)

            if dist_feet >= 0 or dist_center >= 0:
                current_occupants.append(track)
                # Check for staff/waiter visits (recognized employee)
                if track.identity_status == "recognized":
                    has_staff_visit = True

        now = time.time()
        was_occupied = self.occupied
        self.occupant_count = len(current_occupants)
        self.occupant_track_ids = [t.track_id for t in current_occupants]
        self.occupied = (self.occupant_count > 0)

        if self.occupied:
            if not was_occupied or self.first_occupied_time is None:
                self.first_occupied_time = now
            self.current_dwell_sec = max(0.0, now - self.first_occupied_time)

            # Waiter service tracking
            if self.zone_type == "table" and has_staff_visit:
                if self.last_waiter_visit_time is None or (now - self.last_waiter_visit_time > 20.0):
                    self.waiter_visits += 1
                    self.last_waiter_visit_time = now
                    logger.info(f"Waiter service logged for {self.name} on {self.camera_id}")

            # Check for Restricted Intrusion
            if self.zone_type == "restricted":
                self.is_alert = True
                self.alert_message = f"INTRUSION DETECTED: {len(current_occupants)} person(s)"
                self.last_alert_time = now

            # Check for Loitering Anomaly
            elif self.current_dwell_sec > self.max_dwell_sec:
                self.is_alert = True
                self.alert_message = f"LOITERING EXCEEDED: {int(self.current_dwell_sec)}s (Limit: {int(self.max_dwell_sec)}s)"
                self.last_alert_time = now
            else:
                self.is_alert = False
                self.alert_message = ""
        else:
            # Turnover completion
            if was_occupied and self.first_occupied_time is not None:
                dwell = now - self.first_occupied_time
                if dwell >= 3.0: # Minimum filter to avoid transient pass-throughs
                    self.total_visits += 1
                    self.completed_dwell_durations.append(dwell)
                    if len(self.completed_dwell_durations) > 50:
                        self.completed_dwell_durations.pop(0)

            self.first_occupied_time = None
            self.current_dwell_sec = 0.0
            self.is_alert = False
            self.alert_message = ""

    def get_avg_dwell(self) -> float:
        if not self.completed_dwell_durations:
            return round(self.current_dwell_sec, 1) if self.occupied else 0.0
        return round(float(np.mean(self.completed_dwell_durations)), 1)

    def to_dict(self) -> Dict[str, Any]:
        avg_dwell = self.get_avg_dwell()
        # For queue zones: estimated wait time = occupants * (avg_dwell or 60.0s) / 60
        est_wait_min = 0.0
        if self.zone_type == "queue":
            est_wait_min = round((self.occupant_count * 1.5), 1)

        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "name": self.name,
            "zone_type": self.zone_type,
            "polygon": self.polygon,
            "capacity_limit": self.capacity_limit,
            "max_dwell_sec": self.max_dwell_sec,
            "enabled": self.enabled,
            "occupied": self.occupied,
            "occupant_count": self.occupant_count,
            "occupant_track_ids": self.occupant_track_ids,
            "current_dwell_sec": round(self.current_dwell_sec, 1),
            "total_visits": self.total_visits,
            "avg_dwell_sec": avg_dwell,
            "waiter_visits": self.waiter_visits,
            "estimated_wait_min": est_wait_min,
            "is_alert": self.is_alert,
            "alert_message": self.alert_message
        }

class ZoneManager:
    """
    Manages and renders all functional ROI zones for a camera.
    """
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.zones: Dict[str, Zone] = {}
        self._init_default_zones()

    def _init_default_zones(self):
        """Seeds practical, intuitive demo zones if camera matches standard configurations."""
        if self.camera_id in ["CAM-01", "CAM-WEBCAM"]:
            # Entrance / Table / Counter zone
            self.add_zone(Zone(
                zone_id=f"{self.camera_id}-Z1",
                camera_id=self.camera_id,
                name="Front Table A1",
                zone_type="table",
                polygon=[[0.05, 0.35], [0.38, 0.35], [0.38, 0.85], [0.05, 0.85]],
                capacity_limit=2,
                max_dwell_sec=60.0
            ))
            self.add_zone(Zone(
                zone_id=f"{self.camera_id}-Z2",
                camera_id=self.camera_id,
                name="Service Counter Queue",
                zone_type="queue",
                polygon=[[0.55, 0.25], [0.95, 0.25], [0.95, 0.75], [0.55, 0.75]],
                capacity_limit=5,
                max_dwell_sec=45.0
            ))
        elif self.camera_id == "CAM-02":
            # Warehouse / Parking / Restricted zone
            self.add_zone(Zone(
                zone_id=f"{self.camera_id}-Z1",
                camera_id=self.camera_id,
                name="Hazard Machinery Area",
                zone_type="restricted",
                polygon=[[0.60, 0.20], [0.95, 0.20], [0.95, 0.70], [0.60, 0.70]],
                capacity_limit=0,
                max_dwell_sec=5.0
            ))
            self.add_zone(Zone(
                zone_id=f"{self.camera_id}-Z2",
                camera_id=self.camera_id,
                name="Loading Bay Slot P-01",
                zone_type="parking",
                polygon=[[0.10, 0.40], [0.45, 0.40], [0.45, 0.85], [0.10, 0.85]],
                capacity_limit=1,
                max_dwell_sec=120.0
            ))
        elif self.camera_id == "CAM-03":
            # Dining & Lounge tables
            self.add_zone(Zone(
                zone_id=f"{self.camera_id}-Z1",
                camera_id=self.camera_id,
                name="Dining Table #1",
                zone_type="table",
                polygon=[[0.15, 0.30], [0.45, 0.30], [0.45, 0.80], [0.15, 0.80]],
                capacity_limit=4,
                max_dwell_sec=90.0
            ))
            self.add_zone(Zone(
                zone_id=f"{self.camera_id}-Z2",
                camera_id=self.camera_id,
                name="Restricted Server Rack",
                zone_type="restricted",
                polygon=[[0.70, 0.15], [0.95, 0.15], [0.95, 0.65], [0.70, 0.65]],
                capacity_limit=0,
                max_dwell_sec=5.0
            ))

    def add_zone(self, zone: Zone):
        self.zones[zone.id] = zone

    def remove_zone(self, zone_id: str):
        if zone_id in self.zones:
            del self.zones[zone_id]

    def update(self, tracks: List[Any], frame_shape: Tuple[int, int, int]):
        for zone in self.zones.values():
            zone.update(tracks, frame_shape)

    def draw_zones(self, frame: np.ndarray):
        """
        Draws semi-transparent polygon fills and HUD status badges for all active zones.
        """
        if not self.zones:
            return

        h, w = frame.shape[:2]
        overlay = frame.copy()

        for zone in self.zones.values():
            if not zone.enabled:
                continue

            pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in zone.polygon], dtype=np.int32)
            ztype = zone.zone_type

            # Color thematic rules
            if zone.is_alert:
                border_color = (0, 0, 255) # Red flashing
                fill_color = (0, 0, 255)
                alpha = 0.28
            elif ztype == "table":
                border_color = (200, 100, 255) if zone.occupied else (0, 229, 255)
                fill_color = (160, 60, 240) if zone.occupied else (0, 180, 200)
                alpha = 0.20 if zone.occupied else 0.10
            elif ztype == "queue":
                border_color = (0, 171, 255) # Amber
                fill_color = (0, 140, 220)
                alpha = 0.20 if zone.occupied else 0.08
            elif ztype == "parking":
                border_color = (0, 0, 255) if zone.occupied else (0, 230, 118) # Red occupied, Green available
                fill_color = (0, 0, 200) if zone.occupied else (0, 200, 100)
                alpha = 0.22 if zone.occupied else 0.12
            elif ztype == "restricted":
                border_color = (0, 0, 255) if zone.occupied else (0, 100, 255)
                fill_color = (0, 0, 220) if zone.occupied else (0, 60, 180)
                alpha = 0.35 if zone.occupied else 0.12
            else:
                border_color = (180, 180, 180)
                fill_color = (120, 120, 120)
                alpha = 0.10

            # Draw transparent fill
            cv2.fillPoly(overlay, [pts], fill_color)
            cv2.polylines(frame, [pts], True, border_color, 2)

            # Badge Text
            x_min = int(min(p[0] for p in pts))
            y_min = int(min(p[1] for p in pts))
            badge_x = max(10, x_min)
            badge_y = max(24, y_min - 8)

            if zone.is_alert:
                status_text = f"⚠ {zone.name}: {zone.alert_message}"
            elif ztype == "table":
                if zone.occupied:
                    m, s = divmod(int(zone.current_dwell_sec), 60)
                    status_text = f"{zone.name} [OCCUPIED: {m}m {s}s | Visits: {zone.total_visits}]"
                else:
                    status_text = f"{zone.name} [AVAILABLE]"
            elif ztype == "queue":
                wait_min = round(zone.occupant_count * 1.5, 1)
                status_text = f"{zone.name} [WAITING: {zone.occupant_count} | Est: ~{wait_min}m]"
            elif ztype == "parking":
                status_text = f"{zone.name} [{'OCCUPIED' if zone.occupied else 'AVAILABLE'}]"
            elif ztype == "restricted":
                status_text = f"RESTRICTED: {zone.name}"
            else:
                status_text = f"{zone.name} [{zone.occupant_count}]"

            # Badge background
            (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(frame, (badge_x - 4, badge_y - th - 6), (badge_x + tw + 6, badge_y + 4), (15, 18, 25), -1)
            cv2.rectangle(frame, (badge_x - 4, badge_y - th - 6), (badge_x + tw + 6, badge_y + 4), border_color, 1)
            cv2.putText(frame, status_text, (badge_x, badge_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # Apply transparency blend
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    def get_summary(self) -> Dict[str, Any]:
        tables = [z.to_dict() for z in self.zones.values() if z.zone_type == "table"]
        queues = [z.to_dict() for z in self.zones.values() if z.zone_type == "queue"]
        parkings = [z.to_dict() for z in self.zones.values() if z.zone_type == "parking"]
        restricted = [z.to_dict() for z in self.zones.values() if z.zone_type == "restricted"]

        occupied_tables = sum(1 for t in tables if t["occupied"])
        occupied_parking = sum(1 for p in parkings if p["occupied"])
        total_queue_people = sum(q["occupant_count"] for q in queues)
        est_queue_wait = sum(q["estimated_wait_min"] for q in queues)

        return {
            "camera_id": self.camera_id,
            "total_zones": len(self.zones),
            "hospitality": {
                "total_tables": len(tables),
                "occupied_tables": occupied_tables,
                "available_tables": len(tables) - occupied_tables,
                "total_turnovers": sum(t["total_visits"] for t in tables),
                "total_waiter_visits": sum(t["waiter_visits"] for t in tables)
            },
            "queue_analytics": {
                "total_queue_zones": len(queues),
                "total_waiting_people": total_queue_people,
                "estimated_wait_min": round(est_queue_wait, 1)
            },
            "parking": {
                "total_spaces": len(parkings),
                "occupied_spaces": occupied_parking,
                "available_spaces": len(parkings) - occupied_parking,
                "occupancy_rate_pct": round((occupied_parking / len(parkings) * 100), 1) if parkings else 0.0
            },
            "security": {
                "restricted_zones": len(restricted),
                "active_intrusions": sum(1 for r in restricted if r["occupied"]),
                "active_alerts": sum(1 for z in self.zones.values() if z.is_alert)
            },
            "zones": [z.to_dict() for z in self.zones.values()]
        }
