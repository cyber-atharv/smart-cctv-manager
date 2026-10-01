# System Architecture Documentation
# Smart CCTV AI Safety & Identification Platform

## Architectural Overview
The system follows a high-throughput, decoupled, modular micro-pipeline pattern inspired by open-source vision architectures (Frigate, NVIDIA DeepStream, PeekingDuck, and CompreFace).

```
+-----------------------------------------------------------------------------+
|                                Video Ingestion                              |
|   (Webcam / RTSP Stream / MP4 Video File / Synthetic Simulator Stream)      |
+-----------------------------------------------------------------------------+
                                       │
                                       ▼
+-----------------------------------------------------------------------------+
|                            Video Pipeline Worker                            |
|  - In-Memory Circular Rolling Video Buffer (5s pre-event + 10s post-event)  |
|  - Rate-limited frame scheduler                                             |
+-----------------------------------------------------------------------------+
                                       │
                                       ▼
+-----------------------------------------------------------------------------+
|                            Person Detection Layer                           |
|  - YOLOv8 Person Detector (Class 0: person) / OpenCV HOG fallback           |
|  - Confidence & NMS thresholding                                            |
+-----------------------------------------------------------------------------+
                                       │
                                       ▼
+-----------------------------------------------------------------------------+
|                         Multi-Object Tracking Layer                         |
|  - IoU & Centroid Distance Association                                      |
|  - Persistent Track IDs (Track 1, 2, ...)                                   |
|  - Temporal Velocity (vx, vy, speed) & Trajectory History                  |
+-----------------------------------------------------------------------------+
                                       │
              ┌────────────────────────┴────────────────────────┐
              ▼                                                 ▼
+-----------------------------+                 +-----------------------------+
|   Pose & Activity Branch    |                 |   Biometric Face Branch     |
| - MediaPipe Pose Landmarks  |                 | - Upper-body Face Detection |
| - 16 Kinematic Features     |                 | - 128-d L2 Normalized Emb   |
| - Random Forest Classifier  |                 | - Cosine Gallery Match      |
| - Temporal State Machine    |                 | - ID: EMP001 / "Unknown"    |
+-----------------------------+                 +-----------------------------+
              │                                                 │
              └────────────────────────┬────────────────────────┘
                                       │
                                       ▼
+-----------------------------------------------------------------------------+
|                          Event Correlation Engine                           |
|  - Correlated Safety Incident Object:                                       |
|    { event_id, camera_id, track_id, employee_id, activity, conf, ... }      |
|  - Snapshot capture + Asynchronous MP4 incident clip generation             |
+-----------------------------------------------------------------------------+
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
+--------------------+   +---------------------------+   +--------------------+
| SQLite Database    |   | Filesystem Storage        |   | WebSocket Server   |
| (SQLAlchemy ORM)   |   | (/snapshots, /clips)      |   | (Real-time Alerts) |
+--------------------+   +---------------------------+   +--------------------+
                                       │
                                       ▼
+-----------------------------------------------------------------------------+
|                          CCTV Supervisor Dashboard                          |
|  (Live MJPEG Grid, Realtime Incident Alert Modal, Review Confirm/Reject)     |
+-----------------------------------------------------------------------------+
```

## Concurrency & Thread Isolation
- **Camera Ingestion Thread**: Dedicated non-blocking capture thread per camera stream.
- **Vision Pipeline Worker**: Independent thread performing inference and rendering annotated frames at target 15 FPS.
- **Async Clip Writer**: Background thread for compiling rolling ring buffers into standard MP4 incident files upon confirmed fall triggers.
- **FastAPI Async Event Loop**: Handles HTTP REST requests and WebSockets without blocking video inference.
