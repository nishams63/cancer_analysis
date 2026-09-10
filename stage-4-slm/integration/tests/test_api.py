"""
API Test Suite for Stage 4 FastAPI Inference Service.
Tests all endpoints (/health, /generate/risk, /generate/action, /generate/decision-support, /batch),
header latency injection, and input validation bounds.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

INTEG_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(INTEG_SRC_DIR))

from app import app

client = TestClient(app)

SAMPLE_NOTE = (
    "ONCOLOGY PROGRESS NOTE\n"
    "Patient PT-0001 presenting for NSCLC evaluation with EGFR driver mutation.\n"
    "Administer Erlotinib at 150 mg oral daily.\n"
    "Patient tolerating therapy well without acute adverse events."
)


def test_health_endpoint():
    """Verify health endpoint returns 200 and valid metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OK"
    assert "clinical-slm" in data["model_version"]
    assert data["uptime_seconds"] >= 0.0
    assert "X-Process-Time-Ms" in response.headers


def test_generate_risk_endpoint():
    """Verify /v1/generate/risk produces risk attribution."""
    payload = {
        "clinical_note": SAMPLE_NOTE,
        "patient_id": "PT-0001"
    }
    response = client.post("/v1/generate/risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["patient_id"] == "PT-0001"
    assert "target_risk" in data
    assert len(data["target_risk"]) > 10
    assert data["urgency_tier"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def test_generate_action_endpoint():
    """Verify /v1/generate/action produces clinical action recommendations."""
    payload = {
        "clinical_note": SAMPLE_NOTE,
        "patient_id": "PT-0001"
    }
    response = client.post("/v1/generate/action", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["patient_id"] == "PT-0001"
    assert "target_action" in data
    assert data["dose_modification"] in ["HOLD_IMMEDIATELY", "DOSE_REDUCE", "MAINTAIN_CURRENT"]


def test_generate_decision_support_endpoint():
    """Verify primary /v1/generate/decision-support returns complete triad."""
    payload = {
        "document_id": "DOC-999",
        "patient_id": "PT-0001",
        "clinical_note": SAMPLE_NOTE
    }
    response = client.post("/v1/generate/decision-support", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "DOC-999"
    assert "target_risk" in data
    assert "target_key_finding" in data
    assert "target_action" in data
    assert data["safety_gate_passed"] is True
    assert data["routing"] == "STAGE_4_AUTO"
    assert data["latency_ms"] >= 0.0


def test_batch_endpoint():
    """Verify /v1/batch processes batch payloads."""
    payload = {
        "notes": [
            {"document_id": "D1", "patient_id": "P1", "clinical_note": SAMPLE_NOTE},
            {"document_id": "D2", "patient_id": "P2", "clinical_note": SAMPLE_NOTE}
        ]
    }
    response = client.post("/v1/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["results"]) == 2
    assert data["batch_latency_ms"] >= 0.0


def test_input_validation_failure():
    """Verify validation error on malformed short note."""
    response = client.post("/v1/generate/risk", json={"clinical_note": "short", "patient_id": "P1"})
    assert response.status_code == 422  # Unprocessable entity due to min_length constraint
