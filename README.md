# Smart CCTV AI Safety & Identification Platform
### Automated Fall Detection + Activity Recognition + Multi-Object Tracking + Decoupled Face Biometrics + Incident Management

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Tests Passed](https://img.shields.io/badge/tests-15%2F15%20passing-brightgreen.svg)](tests/)

---

## 🌟 Key Highlights & Engineering Features
- **Modular Computer Vision Architecture**: Inspired by NVIDIA DeepStream, Frigate, PeekingDuck, and CompreFace.
- **Safety Decoupling Principle**: Fall detection operates with 100% safety capability regardless of whether a face is recognized or unknown.
- **Multi-Object Tracking (MOT)**: Persistent Track IDs (`Track #7`, `Track #8`) with IoU association and dynamic velocity tracking.
- **16-Feature Kinematic Pose Extractor**: Converts MediaPipe keypoints into invariant postural angles, aspect ratios, and downward velocities.
- **Academic ML Activity Classifier**: Random Forest (99.42% frame-level test accuracy) & Decision Tree (98.00%) on unseen subject benchmark.
- **Temporal Fall State Machine**: 4-stage verification (`NORMAL ➔ SUSPICIOUS ➔ POSSIBLE_FALL ➔ CONFIRMED_FALL`) with floor persistence suppressing raw single-frame false positives.
- **Pluggable Face Biometrics**: Multi-angle enrollment wizard extracting 128-d L2-normalized embeddings via deep convolutional feature backbones.
- **Evidence Management**: Circular ring buffer saving 5s pre-event + 10s post-event MP4 incident clips and JPEG snapshots.
- **Live CCTV Supervisor Dashboard**: Dark glassmorphic interface with live MJPEG streams, WebSocket real-time incident modal, and supervisor confirm/reject actions.
- **Complete Academic Notebooks**: 5 step-by-step Jupyter notebooks for Data Loading, EDA, Feature Engineering, Training, and Evaluation.

---

## 📁 Repository Structure
```
smart-cctv-ai/
├── README.md                      # Project overview & quickstart guide
├── requirements.txt               # Dependencies
├── .env.example                   # Environment configuration template
├── docker-compose.yml             # Container orchestration
│
├── config/                        # YAML-backed system settings
│   ├── system.yaml                # Core storage & server settings
│   ├── cameras.yaml               # Camera registry & sources
│   └── thresholds.yaml            # Fall & CV hyperparameters
│
├── notebooks/                     # Academic ML Workflow Notebooks
│   ├── 01_data_loading.ipynb      # Step 1: Data ingestion & schema inspection
│   ├── 02_eda.ipynb               # Step 2: Exploratory data analysis & plots
│   ├── 03_feature_engineering.ipynb # Step 3: Biomechanical feature formulas
│   ├── 04_model_training.ipynb    # Step 4: Group-aware split & model training
│   └── 05_model_evaluation.ipynb  # Step 5: Confusion matrix & safety metrics
│
├── data/                          # Dataset partitions
│   └── raw/                       # Grouped multi-subject CSV data
│
├── models/                        # Versioned model artifacts
│   ├── fall_detection_model.pkl   # Trained Random Forest classifier
│   ├── feature_columns.json       # 16-feature standardized schema
│   └── metadata.json              # Version, parameters & metrics metadata
│
├── src/                           # Modular source code
│   ├── detection/                 # YOLOv8 & OpenCV person detector
│   ├── tracking/                  # Multi-object tracker & persistent IDs
│   ├── pose/                      # MediaPipe pose & landmark estimator
│   ├── features/                  # 16-feature geometric/kinematic extractor
│   ├── activity/                  # Classifier & temporal state machine
│   ├── face/                      # Face detector, 128-d embedder & enrollment
│   ├── events/                    # Event correlation & rolling clip recorder
│   ├── video/                     # Camera streams, synthetic generator & pipeline
│   ├── database/                  # SQLAlchemy ORM models & database session
│   └── api/                       # FastAPI routes & WebSocket dispatcher
│
├── static/                        # Dashboard assets (CSS, JS, icons)
│   ├── css/styles.css             # Modern dark glassmorphic stylesheet
│   └── js/app.js                  # WebSocket client & alert popup logic
│
├── templates/                     # Jinja2 HTML dashboard templates
│   ├── index.html                 # Main dashboard with live feeds
│   ├── cameras.html               # Fleet stream grid
│   ├── events.html                # Incident log & review panel
│   ├── employees.html             # Employee registry
│   ├── enrollment.html            # Multi-angle face registration wizard
│   ├── analytics.html             # Academic ML evaluation report
│   └── settings.html              # Thresholds & system config
│
├── tests/                         # Test suite
│   ├── test_features.py           # Feature extraction unit tests
│   ├── test_classifier.py         # Activity classification tests
│   ├── test_temporal_verifier.py  # State machine transition tests
│   ├── test_tracker.py            # Tracking & biometric tests
│   └── test_api.py                # FastAPI endpoint integration tests
│
└── docs/                          # Comprehensive Academic Documentation
    ├── academic_report.md         # Full college mini-project report
    ├── architecture.md            # System architecture & component diagram
    ├── ml_pipeline.md             # Kinematic features & leakage prevention
    ├── face_enrollment.md         # Biometric architecture & privacy rules
    └── deployment.md              # Local & Colab setup instructions
```

---

## 🚀 Quickstart Guide

### 1. Installation
```bash
git clone <repo-url>
cd "smart cctv manager"
pip install -r requirements.txt
```

### 2. Train ML Model
```bash
python -m src.activity.train
```

### 3. Run Unit & Integration Tests
```bash
python -m pytest -v
```

### 4. Start the CCTV Platform
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

### 5. Access the Web Dashboard
Open `http://127.0.0.1:8000` in your web browser.

---

## 🧪 Demonstration Scenarios
1. **Registered Employee Walking ➔ Fall Incident**:
   - Camera detects person ➔ Track 1 created ➔ Face recognized as `EMP001 (Rahul Sharma)`.
   - Person trips and falls ➔ Downward velocity detected ➔ Enters `SUSPICIOUS` ➔ `POSSIBLE_FALL` ➔ Floor persistence confirms fall.
   - `CONFIRMED_FALL` triggers event `EVT-...` ➔ High-priority alert banner & audio alert pop up on dashboard with snapshot and employee badge ➔ Supervisor clicks **[Confirm Fall]**.
2. **Unregistered / Unknown Person Fall Incident**:
   - Unknown subject in zone ➔ Face is unrecognized (`identity_status = "unknown"`).
   - Fall occurs ➔ Temporal state machine confirms fall ➔ Valid safety event dispatched with `Unknown Person` tag.
   - **Proves complete safety decoupling from biometric recognition.**
3. **Transient Motion False Alarm Prevention**:
   - Person rapidly bends over to tie shoes or pick up a tool.
   - Stage transitions to `SUSPICIOUS`, but before floor persistence count is reached, person resumes normal posture.
   - State machine resets to `NORMAL` without triggering false alarms.
