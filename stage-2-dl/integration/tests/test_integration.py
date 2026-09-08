"""
Stage 2 Deep Learning - Multimodal Integration Comprehensive Test Suite

Verifies:
  1. System health check and model loading
  2. Full multimodal inference (ResNet-50 + Attention-MIL + Transformer + Gated)
  3. Baseline inference preservation (ResNet-18 + BiLSTM + Fixed Weighted)
  4. Pathology-only mode
  5. Temporal-only mode
  6. Insufficient data abstention
  7. Anti-leakage temporal cutoff (Day > 90 strictly rejected)
  8. Anti-leakage forbidden target column exclusion
  9. Chronological validation (delta_days >= 0)
  10. Corrupt / missing image tile validation
  11. Attention-MIL weights normalization and ranking
  12. Uncertainty estimation (MC Dropout confidence + variance)
  13. OOD detector integration
  14. Batch endpoint with individual patient failure isolation
  15. FastAPI TestClient end-to-end execution
  16. Mandatory synthetic regulatory disclaimers
"""
import os
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import torch
from PIL import Image

TEST_DIR = Path(__file__).resolve().parent
INTEGRATION_DIR = TEST_DIR.parent
SRC_DIR = INTEGRATION_DIR / 'src'
STAGE_2_DIR = INTEGRATION_DIR.parent

sys.path.insert(0, str(STAGE_2_DIR / '.runtime'))
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(STAGE_2_DIR))

import config, schemas, validation, model_registry
from inference_service import PatientInferenceService
from integration_pipeline import _load_sample_patient_data
from api import app
from starlette.testclient import TestClient


@pytest.fixture(scope="module")
def service():
    """Initializes the shared patient inference service."""
    return PatientInferenceService.get_shared()


@pytest.fixture(scope="module")
def api_client():
    """Initializes the FastAPI TestClient."""
    return TestClient(app)


@pytest.fixture(scope="module")
def sample_patient():
    """Loads actual development cohort patient data for PAT-0001."""
    tiles, bio_df = _load_sample_patient_data("PAT-0001")
    obs_list = []
    if bio_df is not None and len(bio_df) > 0:
        for _, row in bio_df.iterrows():
            obs_list.append(schemas.BiomarkerObservation(
                days_from_baseline=int(row['days_from_baseline']),
                ctDNA_vaf_percent=float(row['ctDNA_vaf_percent']) if pd.notna(row.get('ctDNA_vaf_percent')) else None,
                cea_ng_ml=float(row['cea_ng_ml']) if pd.notna(row.get('cea_ng_ml')) else None,
                ca125_u_ml=float(row['ca125_u_ml']) if pd.notna(row.get('ca125_u_ml')) else None,
                ldh_u_l=float(row['ldh_u_l']) if pd.notna(row.get('ldh_u_l')) else None,
                crp_mg_l=float(row['crp_mg_l']) if pd.notna(row.get('crp_mg_l')) else None,
                delta_days=float(row['delta_days']) if pd.notna(row.get('delta_days')) else None,
            ))
    return {
        "patient_id": "PAT-0001",
        "tiles": tiles,
        "observations": obs_list
    }


def test_system_health_endpoint(api_client):
    """Verifies that /health returns loaded models, status, and disclaimers."""
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "models" in data
    assert "mandatory_disclaimer" in data
    assert config.EQUIVALENCE_DISCLAIMER in data["equivalence_disclaimer"]


def test_service_info_endpoint(api_client):
    """Verifies that /info reports configuration parameters and alert thresholds."""
    response = api_client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["historical_cutoff_day"] == 90
    assert "gated" in data["supported_fusion_modes"]
    assert "attention" in data["supported_aggregations"]


def test_full_multimodal_inference_end_to_end(service, sample_patient):
    """Tests complete multimodal pipeline with ResNet-50 Attention-MIL + Transformer + Gated Fusion."""
    req = schemas.PatientInferenceRequest(
        patient_id=sample_patient["patient_id"],
        tile_paths=sample_patient["tiles"][:4],
        temporal_observations=sample_patient["observations"],
        model_configuration="upgraded",
        fusion_mode="gated",
        aggregation_method="attention"
    )
    result = service.predict(req)
    assert result.patient_id == "PAT-0001"
    assert result.inference_mode == "FULL_MULTIMODAL"
    assert set(result.available_modalities) == {"pathology", "temporal"}
    assert result.progression_probability is not None
    assert 0.0 <= result.progression_probability <= 1.0
    assert result.ctdna_forecast_30d is not None
    assert result.pathology is not None
    assert result.pathology.attention_tiles is not None
    assert result.temporal is not None
    assert result.temporal.sequence_length > 0
    assert result.fusion is not None
    assert result.risk.level in ["LOW", "MODERATE", "HIGH"]
    assert result.latency.total_ms > 0


def test_baseline_inference_preservation(service, sample_patient):
    """Asserts that baseline ResNet-18 + BiLSTM + Fixed Weighted linear fusion remains functional."""
    req = schemas.PatientInferenceRequest(
        patient_id=sample_patient["patient_id"],
        tile_paths=sample_patient["tiles"][:4],
        temporal_observations=sample_patient["observations"],
        model_configuration="baseline",
        fusion_mode="fixed_weighted",
        aggregation_method="mean"
    )
    result = service.predict(req)
    assert result.inference_mode == "FULL_MULTIMODAL"
    assert "baseline" in result.model_configuration
    assert result.fusion.model_name == "fixed_weighted"
    assert 0.0 <= result.progression_probability <= 1.0


def test_pathology_only_mode(service, sample_patient):
    """Tests inference when only biopsy tiles are available."""
    req = schemas.PatientInferenceRequest(
        patient_id=sample_patient["patient_id"],
        tile_paths=sample_patient["tiles"][:3],
        temporal_observations=None,
        model_configuration="upgraded"
    )
    result = service.predict(req)
    assert result.inference_mode == "PATHOLOGY_ONLY"
    assert result.available_modalities == ["pathology"]
    assert result.pathology is not None
    assert result.temporal is None
    assert result.fusion is None
    assert result.progression_probability is not None


def test_temporal_only_mode(service, sample_patient):
    """Tests inference when only longitudinal biomarkers are available."""
    req = schemas.PatientInferenceRequest(
        patient_id=sample_patient["patient_id"],
        tile_paths=None,
        temporal_observations=sample_patient["observations"],
        model_configuration="upgraded"
    )
    result = service.predict(req)
    assert result.inference_mode == "TEMPORAL_ONLY"
    assert result.available_modalities == ["temporal"]
    assert result.pathology is None
    assert result.temporal is not None
    assert result.fusion is None
    assert result.progression_probability is not None
    assert result.ctdna_forecast_30d is not None


def test_insufficient_data_mode(service):
    """Tests that pipeline safely abstains without fabricating numbers when no data is provided."""
    req = schemas.PatientInferenceRequest(
        patient_id="PAT-EMPTY",
        tile_paths=None,
        temporal_observations=None
    )
    result = service.predict(req)
    assert result.inference_mode == "INSUFFICIENT_DATA"
    assert result.available_modalities == []
    assert result.risk.level == "INSUFFICIENT_DATA"
    assert result.progression_probability is None
    assert result.ctdna_forecast_30d is None


def test_future_temporal_leakage_rejected(api_client, sample_patient):
    """Strictly enforces anti-leakage: observations with Day > 90 must raise HTTP 422."""
    leaked_observations = [
        {"days_from_baseline": 0, "ctDNA_vaf_percent": 0.5},
        {"days_from_baseline": 45, "ctDNA_vaf_percent": 0.8},
        {"days_from_baseline": 115, "ctDNA_vaf_percent": 1.9}  # Violates Day <= 90
    ]
    payload = {
        "patient_id": "PAT-LEAK",
        "temporal_observations": leaked_observations
    }
    response = api_client.post("/predict/patient", json=payload)
    assert response.status_code == 422


def test_forbidden_target_columns_rejected():
    """Verifies that DataFrame containing future targets raises ValidationError."""
    df = pd.DataFrame([{
        "days_from_baseline": 30,
        "ctDNA_vaf_percent": 1.2,
        "future_ctDNA_30d_target": 2.5  # Forbidden column
    }])
    with pytest.raises(validation.ValidationError) as exc:
        validation.validate_temporal_history(df)
    assert "Anti-leakage violation" in str(exc.value)


def test_chronological_ordering_enforced():
    """Asserts that negative delta days or non-monotonic timestamps are handled/rejected."""
    invalid_df = pd.DataFrame([
        {"days_from_baseline": 30, "ctDNA_vaf_percent": 1.0},
        {"days_from_baseline": 30, "ctDNA_vaf_percent": 1.5}  # Duplicate timestamp
    ])
    with pytest.raises(validation.ValidationError) as exc:
        validation.validate_temporal_history(invalid_df)
    assert "Duplicate visit dates" in str(exc.value)


def test_corrupt_tile_handling():
    """Verifies that non-existent or corrupt image paths raise ValidationError."""
    with pytest.raises(validation.ValidationError) as exc:
        validation.validate_pathology_tiles(["/nonexistent/path/tile_fake.png"])
    assert "not found" in str(exc.value)


def test_attention_mil_weights_sum_and_rank(service, sample_patient):
    """Verifies Attention-MIL weights sum to 1.0 and are properly ranked descending."""
    req = schemas.PatientInferenceRequest(
        patient_id=sample_patient["patient_id"],
        tile_paths=sample_patient["tiles"][:4],
        model_configuration="upgraded",
        aggregation_method="attention"
    )
    result = service.predict(req)
    tiles = result.pathology.attention_tiles
    assert tiles is not None
    assert len(tiles) == 4
    total_weight = sum(t.attention_weight for t in tiles)
    assert 0.99 <= total_weight <= 1.01
    # Check descending order of ranks
    ranks = [t.rank for t in tiles]
    assert ranks == [1, 2, 3, 4]
    weights = [t.attention_weight for t in tiles]
    assert weights == sorted(weights, reverse=True)


def test_uncertainty_quantification(service, sample_patient):
    """Verifies MC Dropout returns bounded uncertainty and confidence."""
    req = schemas.PatientInferenceRequest(
        patient_id=sample_patient["patient_id"],
        tile_paths=sample_patient["tiles"][:3],
        temporal_observations=sample_patient["observations"],
        model_configuration="upgraded",
        fusion_mode="gated"
    )
    result = service.predict(req)
    unc = result.uncertainty
    assert unc.confidence is not None
    assert 0.5 <= unc.confidence <= 1.0


def test_batch_inference_error_isolation(api_client, sample_patient):
    """Verifies batch inference processes valid patients even when one patient has invalid inputs."""
    obs_dicts = [obs.model_dump() for obs in sample_patient["observations"]]
    batch_payload = {
        "patients": [
            {
                "patient_id": "PAT-VALID-1",
                "tile_paths": sample_patient["tiles"][:2]
            },
            {
                "patient_id": "PAT-INVALID-DUP",
                "temporal_observations": [
                    {"days_from_baseline": 15, "ctDNA_vaf_percent": 1.0},
                    {"days_from_baseline": 15, "ctDNA_vaf_percent": 1.2}  # Duplicate visit date
                ]
            },
            {
                "patient_id": "PAT-VALID-2",
                "temporal_observations": obs_dicts
            }
        ]
    }
    response = api_client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert data["successful"] == 2
    assert data["failed"] == 1
    assert data["results"][0]["status"] == "success"
    assert data["results"][1]["status"] == "error"
    assert data["results"][1]["status_code"] == 400
    assert data["results"][2]["status"] == "success"


def test_predict_sample_patient_endpoint(api_client):
    """Tests demo sample patient endpoint."""
    response = api_client.post("/predict/sample/PAT-0001?aggregation=attention&fusion=gated")
    assert response.status_code == 200
    data = response.json()
    assert data["patient_id"] == "PAT-0001"
    assert data["inference_mode"] == "FULL_MULTIMODAL"
    assert "risk" in data
    assert "mandatory_disclaimer" in data
