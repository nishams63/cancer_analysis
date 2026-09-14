"""Tests for FastAPI HTTP endpoints."""
import pytest
from fastapi.testclient import TestClient
from api.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["registered_tools_count"] >= 15


def test_run_workflow_endpoint(client):
    resp = client.post("/agent/run", json={"intent": "revenue_decline"})
    assert resp.status_code == 200
    data = resp.json()
    assert "run_id" in data
    assert "confidence" in data
    assert "status" in data


def test_get_run_and_trace_endpoints(client):
    run_resp = client.post("/agent/run", json={"intent": "revenue_decline"})
    run_id = run_resp.json()["run_id"]

    state_resp = client.get(f"/agent/runs/{run_id}")
    assert state_resp.status_code == 200
    assert state_resp.json()["run_id"] == run_id

    trace_resp = client.get(f"/agent/runs/{run_id}/trace")
    assert trace_resp.status_code == 200
    assert isinstance(trace_resp.json(), list)
    assert len(trace_resp.json()) >= 1

    status_resp = client.get(f"/agent/runs/{run_id}/status")
    assert status_resp.status_code == 200
    assert "status" in status_resp.json()


def test_override_endpoint(client):
    run_resp = client.post("/agent/run", json={"intent": "revenue_decline"})
    run_id = run_resp.json()["run_id"]

    override_resp = client.post(
        f"/agent/runs/{run_id}/override",
        json={"decision": "approve", "reason": "Analyst manual signoff"},
    )
    assert override_resp.status_code == 200
    assert override_resp.json()["status"] == "success"
