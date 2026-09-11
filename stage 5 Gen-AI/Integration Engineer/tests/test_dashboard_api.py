from src.dashboard import create_app
from fastapi.testclient import TestClient

def test_dashboard_endpoints(temp_db):
    app = create_app(temp_db)
    client = TestClient(app)

    # Health
    r_h = client.get("/api/stage5/health")
    assert r_h.status_code == 200
    assert "status" in r_h.json()

    # Summary
    r_s = client.get("/api/stage5/summary")
    assert r_s.status_code == 200
    assert "total_scenarios" in r_s.json()

    # Scenarios list
    r_scen = client.get("/api/stage5/scenarios")
    assert r_scen.status_code == 200
    assert isinstance(r_scen.json(), list)
