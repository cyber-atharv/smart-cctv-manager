# Deployment & Operation Guide

## 1. Local Quickstart (Windows / Linux / macOS)

### Prerequisites
- Python 3.10+
- Webcam or standard video source (Synthetic generator included by default)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train & Export ML Models (Optional, Pre-trained artifacts included)
```bash
python -m src.activity.train
```

### Step 3: Launch CCTV Safety Platform Server
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

### Step 4: Access Web Dashboard
Open your web browser and navigate to:
```
http://127.0.0.1:8000
```

---

## 2. Running in Google Colab
1. Upload the `notebooks/` directory to Google Drive / Colab.
2. Open `04_model_training.ipynb` or `05_model_evaluation.ipynb`.
3. Select Runtime: **Python 3 CPU or T4 GPU**.
4. Run all cells to train the Random Forest / Decision Tree model and export `fall_detection_model.pkl`.

---

## 3. Connecting Real RTSP Cameras or Physical Webcams
Edit `config/cameras.yaml`:
```yaml
cameras:
  - id: "CAM-01"
    name: "Factory Floor Entrance"
    location: "Building 1 - Door A"
    source_type: "webcam" # Use 'webcam', 'rtsp', or 'file'
    source_uri: "0"       # Webcam index '0' or RTSP URI 'rtsp://user:pass@192.168.1.50:554/h264'
    fps: 15
    enabled: true
```
Save the file and restart the server or add cameras via the API.
