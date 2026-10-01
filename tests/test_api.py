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
    for path in ["/cameras", "/events", "/employees", "/enrollment", "/analytics", "/settings"]:
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
    assert data["status"] == "healthy"
    assert "ml_model" in data

def test_api_events_simulate_and_review():
    client = TestClient(app)
    
    # 1. Simulate Fall Event
    sim_res = client.post("/api/events/simulate?camera_id=CAM-01&employee_code=EMP001")
    assert sim_res.status_code == 200
    evt_data = sim_res.json()["event"]
    event_id = evt_data["id"]
    assert evt_data["activity"] == "falling"
    assert evt_data["employee_code"] == "EMP001"

    # 2. Confirm Event
    conf_res = client.post(f"/api/events/{event_id}/confirm?reviewer=TestSupervisor")
    assert conf_res.status_code == 200
    assert conf_res.json()["event"]["status"] == "confirmed"
