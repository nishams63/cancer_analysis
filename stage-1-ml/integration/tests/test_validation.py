"""
Validation and boundary tests for request payloads and schema integrity.
Role: Integration Engineer.

Covers:
- TEST 6: Missing required field rejection (HTTP 422)
- TEST 7: Invalid data type rejection (HTTP 422)
- TEST 8: Unknown categorical value handling (OneHotEncoder handle_unknown='ignore')
- Out-of-bounds numerical values validation
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
def base_payload():
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


def test_6_missing_required_field(client, base_payload):
    """TEST 6: Send an input with a required field missing (e.g. hemoglobin). Verify 422 validation failure."""
    bad_payload = dict(base_payload)
    del bad_payload["hemoglobin"]

    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422
    data = resp.json()
    assert "error" in data
    assert data["error"] == "Input validation failed"
    # Ensure the details explicitly mention the missing field
    details_str = " ".join(data.get("details", []))
    assert "hemoglobin" in details_str


def test_missing_multiple_required_fields(client, base_payload):
    """Verify multiple missing fields are all reported in the validation error details."""
    bad_payload = dict(base_payload)
    del bad_payload["age"]
    del bad_payload["cancer_type"]
    del bad_payload["drug_dose"]

    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422
    data = resp.json()
    details_str = " ".join(data.get("details", []))
    assert "age" in details_str
    assert "cancer_type" in details_str
    assert "drug_dose" in details_str


def test_7_invalid_data_type(client, base_payload):
    """TEST 7: Send an invalid data type (e.g. string for numeric age). Verify 422 validation failure."""
    bad_payload = dict(base_payload)
    bad_payload["age"] = "not_a_valid_number"

    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422
    data = resp.json()
    assert data["error"] == "Input validation failed"
    details_str = " ".join(data.get("details", []))
    assert "age" in details_str


def test_invalid_negative_numeric_values(client, base_payload):
    """Verify out-of-range numerical values (e.g. negative age or dose) are rejected by schema."""
    bad_payload = dict(base_payload)
    bad_payload["age"] = -5.0

    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422
    data = resp.json()
    details_str = " ".join(data.get("details", []))
    assert "age" in details_str


def test_8_unknown_categorical_values(client, base_payload):
    """
    TEST 8: Test unknown categorical values according to existing preprocessing behavior.
    The frozen OneHotEncoder uses handle_unknown='ignore', so novel categories do not crash,
    but encode to zero-vectors for that categorical column.
    """
    novel_payload = dict(base_payload)
    novel_payload["cancer_type"] = "ExtremelyRareUnseenTumorType"
    novel_payload["drug_name"] = "NovelExperimentalCompoundX"
    novel_payload["mutation_primary"] = "UnknownRareVariantQ"

    resp = client.post("/predict", json=novel_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["predicted_toxicity_risk"] in ["Low", "Moderate", "High"]
    assert pytest.approx(sum(data["probabilities"].values()), abs=1e-3) == 1.0


def test_boolean_string_coercion(client, base_payload):
    """Verify boolean strings ('true', 'yes', '1') for previous_adverse_event are coerced properly."""
    payload_str_bool = dict(base_payload)
    payload_str_bool["previous_adverse_event"] = "true"

    resp = client.post("/predict", json=payload_str_bool)
    assert resp.status_code == 200
    assert resp.json()["predicted_toxicity_risk"] in ["Low", "Moderate", "High"]
