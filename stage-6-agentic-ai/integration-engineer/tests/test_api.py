"""Tests for FastAPI REST endpoints."""
from fastapi.testclient import TestClient


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


def test_start_run_endpoint(client):
    res = client.post("/api/v1/aada/run", json={"goal": "Analyze why revenue decreased last quarter", "sync": True})
    assert res.status_code == 200
    data = res.json()
    assert "run_id" in data
    assert data["status"] in ("COMPLETED", "WAITING_FOR_HUMAN", "RUNNING")


def test_status_endpoint(client):
    res = client.post("/api/v1/aada/run", json={"goal": "Analyze why revenue decreased last quarter", "sync": True})
    run_id = res.json()["run_id"]

    stat_res = client.get(f"/api/v1/aada/runs/{run_id}/status")
    assert stat_res.status_code == 200
    stat_data = stat_res.json()
    assert stat_data["run_id"] == run_id
    assert "progress" in stat_data


def test_workflow_and_result_endpoints(client):
    res = client.post("/api/v1/aada/run", json={"goal": "Analyze why revenue decreased last quarter", "sync": True})
    run_id = res.json()["run_id"]

    wf_res = client.get(f"/api/v1/aada/runs/{run_id}/workflow")
    assert wf_res.status_code == 200
    assert "workflow_id" in wf_res.json()

    res_res = client.get(f"/api/v1/aada/runs/{run_id}/result")
    assert res_res.status_code == 200
    assert "summary" in res_res.json()


def test_invalid_request_handling(client):
    res = client.post("/api/v1/aada/run", json={"goal": ""})
    assert res.status_code in (400, 422)


def test_not_found_run_handling(client):
    res = client.get("/api/v1/aada/runs/RUN-DOES-NOT-EXIST")
    assert res.status_code == 404
    data = res.json()
    assert data["error"]["code"] == "RUN_NOT_FOUND"


def test_human_approval_api_endpoints(client, session_manager):
    session = session_manager.create_session(run_id="RUN-API-APP", goal="Analyze churn")
    session.human_review.required = True
    session.human_review.status = "PENDING"
    session_manager.save_session(session)

    res = client.post("/api/v1/aada/runs/RUN-API-APP/approve", json={"reason": "Approved in UI"})
    assert res.status_code == 200
    assert res.json()["status"] == "resumed"
