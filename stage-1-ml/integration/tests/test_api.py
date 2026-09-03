"""
Integration tests for FastAPI inference endpoints (/health, /predict, /predict/batch).
Role: Integration Engineer.

Covers:
- TEST 4: Valid input prediction format ("Low" | "Moderate" | "High")
- TEST 5: Calibrated probability structure and normalization
- Endpoint health and batch functionality
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Path configuration
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_payload():
    return {
        "age": 62.0,
        "sex": "Female",
        "cancer_type": "NSCLC",
        "cancer_stage": "Stage III",
        "smoking_history": "Former",
        "mutation_primary": "EGFR",
        "mutation_secondary": "TP53",
        "mutation_burden": 6.8,
        "gene_expression_score": 49.8,
        "ctdna_level": 1.4,
        "tumor_marker_level": 24.7,
        "inflammation_marker": 4.2,
        "biomarker_trend": "Stable",
        "heart_rate": 79.0,
        "systolic_bp": 128.0,
        "diastolic_bp": 80.0,
        "oxygen_saturation": 96.0,
        "hemoglobin": 12.5,
        "white_blood_cell_count": 7.41,
        "platelet_count": 231.0,
        "creatinine_level": 1.0,
        "liver_function_marker": 17.3,
        "treatment_type": "Targeted Therapy",
        "drug_name": "Erlotinib",
        "drug_dose": 150.0,
        "treatment_cycle": 4,
        "previous_treatment_count": 1,
        "previous_adverse_event": False,
        "previous_toxicity_grade": 0.0,
        "comorbidity_count": 1.0
    }


def test_root_endpoint(client):
    """Verify root info endpoint returns 200 with service metadata."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "Oncology Toxicity Risk Prediction Service"
    assert data["model_version"] == "V4"
    assert data["status"] == "online"


def test_health_endpoint(client):
    """TEST 10 (Health): Verify /health endpoint returns 200, status 'ok', and model_version 'V4'."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_version"] == "V4"
    assert data["artifacts_loaded"] is True
    assert "timestamp" in data


def test_4_predict_endpoint_valid_class(client, valid_payload):
    """TEST 4: Send one valid input and verify prediction is Low OR Moderate OR High."""
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_toxicity_risk" in data
    assert data["predicted_toxicity_risk"] in ["Low", "Moderate", "High"]
    assert data["model_version"] == "V4"
    assert "disclaimer" in data


def test_5_predict_endpoint_probabilities(client, valid_payload):
    """TEST 5: Verify probabilities are present, numeric, within [0, 1], and sum to ~1.0."""
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "probabilities" in data
    probs = data["probabilities"]

    # All classes present
    assert set(probs.keys()) == {"Low", "Moderate", "High"}

    # Numeric and bounded [0, 1]
    for class_name, prob_val in probs.items():
        assert isinstance(prob_val, (int, float))
        assert 0.0 <= prob_val <= 1.0

    # Sum to 1.0
    prob_sum = sum(probs.values())
    assert pytest.approx(prob_sum, abs=1e-3) == 1.0


def test_predict_batch_endpoint(client, valid_payload):
    """Verify /predict/batch endpoint accepts multiple records and returns structured response."""
    rec2 = dict(valid_payload)
    rec2["age"] = 72.0
    rec2["creatinine_level"] = 2.4
    rec2["cancer_stage"] = "Stage IV"

    resp = client.post("/predict/batch", json=[valid_payload, rec2])
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_records"] == 2
    assert data["model_version"] == "V4"
    assert len(data["predictions"]) == 2

    for pred in data["predictions"]:
        assert pred["predicted_toxicity_risk"] in ["Low", "Moderate", "High"]
        assert pytest.approx(sum(pred["probabilities"].values()), abs=1e-3) == 1.0
