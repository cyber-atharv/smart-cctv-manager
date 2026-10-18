import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.database.db import init_db

@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    init_db()

def test_dashboard_root_html_route():
    """Regression test ensuring GET / renders HTTP 200 without TemplateResponse type errors."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Smart CCTV AI Safety Platform" in response.text
    assert "Facility Operations & Safety Dashboard" in response.text

def test_dashboard_html_subpages():
    """Verifies all dashboard UI subpages return HTTP 200."""
    client = TestClient(app)
    for path in ["/cameras", "/people-intelligence", "/zone-analytics", "/events", "/employees", "/enrollment", "/analytics", "/settings"]:
        response = client.get(path)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

def test_api_cameras():
    client = TestClient(app)
    response = client.get("/api/cameras")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert "id" in data[0]
    assert "source_type" in data[0]

def test_api_camera_verification():
    """Tests the multi-step real-time camera verification pipeline."""
    client = TestClient(app)
    # Test demo stream verification
    res_demo = client.post("/api/cameras/test", json={"source_type": "demo", "source_uri": "synthetic"})
    assert res_demo.status_code == 200
    data_demo = res_demo.json()
    assert data_demo["success"] is True
    assert len(data_demo["steps"]) == 5

    # Test local webcam verification
    res_cam = client.post("/api/cameras/test", json={"source_type": "webcam", "source_uri": "0"})
    assert res_cam.status_code == 200
    data_cam = res_cam.json()
    assert "steps" in data_cam

def test_api_camera_scan():
    client = TestClient(app)
    res = client.post("/api/cameras/scan-local")
    assert res.status_code == 200
    assert "devices" in res.json()

def test_api_people_intelligence():
    client = TestClient(app)
    res = client.get("/api/people-intelligence")
    assert res.status_code == 200
    data = res.json()
    assert "subjects" in data
    assert "total_active_people" in data

def test_api_employees():
    client = TestClient(app)
    response = client.get("/api/employees")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1

def test_api_system_status():
    client = TestClient(app)
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "pose_engine" in data

def test_api_events_simulate_and_review():
    client = TestClient(app)
    
    # 1. Simulate Fall Event
    sim_res = client.post("/api/events/simulate?camera_id=CAM-01")
    assert sim_res.status_code == 200
    evt_data = sim_res.json()["event"]
    event_id = evt_data["id"]
    assert evt_data["activity"] == "falling"

    # 2. Confirm Event
    conf_res = client.post(f"/api/events/{event_id}/confirm?reviewer=TestSupervisor")
    assert conf_res.status_code == 200
    assert conf_res.json()["event"]["status"] == "confirmed"

def test_api_multi_domain_endpoints():
    client = TestClient(app)

    # 1. Test GET /api/zones
    res_zones = client.get("/api/zones")
    assert res_zones.status_code == 200
    data_zones = res_zones.json()
    assert "cameras" in data_zones

    # 2. Test POST & DELETE /api/zones on CAM-01
    add_zone_payload = {
        "camera_id": "CAM-01",
        "id": "TEST-ZONE-99",
        "name": "Test Dining Table",
        "zone_type": "table",
        "polygon": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.4], [0.1, 0.4]],
        "capacity_limit": 4,
        "max_dwell_sec": 45.0
    }
    res_add = client.post("/api/zones", json=add_zone_payload)
    assert res_add.status_code == 200
    assert res_add.json()["success"] is True

    # Delete the test zone
    res_del = client.delete("/api/zones/CAM-01/TEST-ZONE-99")
    assert res_del.status_code == 200
    assert res_del.json()["success"] is True

    # 3. Test GET /api/ppe/status
    res_ppe = client.get("/api/ppe/status")
    assert res_ppe.status_code == 200
    data_ppe = res_ppe.json()
    assert "compliance_rate_pct" in data_ppe

    # 4. Test GET /api/hospitality/status
    res_hosp = client.get("/api/hospitality/status")
    assert res_hosp.status_code == 200
    data_hosp = res_hosp.json()
    assert "total_tables" in data_hosp
    assert "occupied_tables" in data_hosp

    # 5. Test GET /api/queue/status
    res_queue = client.get("/api/queue/status")
    assert res_queue.status_code == 200
    data_queue = res_queue.json()
    assert "total_waiting_people" in data_queue

    # 6. Test GET /api/parking/status
    res_parking = client.get("/api/parking/status")
    assert res_parking.status_code == 200
    data_parking = res_parking.json()
    assert "total_spaces" in data_parking
    assert "available_spaces" in data_parking

