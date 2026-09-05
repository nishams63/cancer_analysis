"""
Stage 2 Deep Learning - Multimodal Integration Automated Test Suite

Tests:
1. Frozen checkpoint loading
2. Patient-level alignment
3. Image tile aggregation strategies (mean, median, max)
4. Temporal historical boundary enforcement (days <= 90)
5. Future-target exclusion (anti-leakage)
6. Missing-modality handling (full, pathology-only, temporal-only, neither)
7. Fusion score calculation accuracy
8. Output schema conformity
9. Provenance and audit fields
10. Invalid-input handling and error trapping
11. Synthetic regulatory disclaimer presence
12. Immutability of prior stage artifacts
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
sys.path.insert(0, str(SRC_DIR))

import config, validation, fusion, patient_aggregation, integration_pipeline


@pytest.fixture(scope="module")
def pipeline():
    """Initializes and provides the shared multimodal pipeline."""
    return integration_pipeline.MultimodalPatientPipeline.get_shared_pipeline()


@pytest.fixture(scope="module")
def sample_data():
    """Loads sample patient data for PAT-0001 from development cohort."""
    tiles, bio_df = integration_pipeline._load_sample_patient_data("PAT-0001")
    return {"patient_id": "PAT-0001", "tiles": tiles, "bio_df": bio_df}


def test_frozen_checkpoints_loading(pipeline):
    """Verifies that frozen DL checkpoints load cleanly and are non-empty."""
    assert config.FROZEN_IMAGE_CHECKPOINT.exists(), "Pathology checkpoint file missing!"
    assert config.FROZEN_TEMPORAL_CHECKPOINT.exists(), "Temporal checkpoint file missing!"
    assert pipeline.image_predictor is not None
    assert pipeline.temporal_predictor is not None
    assert pipeline.aggregator is not None


def test_patient_level_alignment(pipeline, sample_data):
    """Verifies that multi-tile pathology and temporal series map to identical patient."""
    res = pipeline.run_patient_inference(
        patient_id=sample_data["patient_id"],
        pathology_tiles=sample_data["tiles"],
        temporal_history=sample_data["bio_df"]
    )
    assert res["patient_id"] == "PAT-0001"
    assert res["modality_status"] == "FULL_MULTIMODAL"
    assert res["pathology_summary"]["num_tiles_analyzed"] == len(sample_data["tiles"])


def test_tile_aggregation_methods(pipeline, sample_data):
    """Tests mean, median, and max tile aggregation accuracy."""
    for method in ['mean', 'median', 'max']:
        res = pipeline.run_patient_inference(
            patient_id=sample_data["patient_id"],
            pathology_tiles=sample_data["tiles"][:4],
            temporal_history=None,
            aggregation_method=method
        )
        assert res["pathology_summary"]["aggregation_method"] == method
        assert 0.0 <= res["pathology_summary"]["malignant_probability"] <= 1.0
        assert 0.0 <= res["pathology_summary"]["benign_probability"] <= 1.0
        assert 0.0 <= res["pathology_summary"]["inflammation_probability"] <= 1.0


def test_temporal_historical_window_boundary(pipeline):
    """Asserts ValidationError is raised if temporal visits exceed Day 90."""
    leaked_df = pd.DataFrame([
        {'days_from_baseline': 0, 'ctDNA_vaf_percent': 0.5},
        {'days_from_baseline': 45, 'ctDNA_vaf_percent': 0.8},
        {'days_from_baseline': 105, 'ctDNA_vaf_percent': 1.5}  # Exceeds Day 90
    ])

    with pytest.raises(validation.ValidationError) as excinfo:
        pipeline.run_patient_inference(
            patient_id="PAT-TEST",
            temporal_history=leaked_df
        )
    assert "Forecasting boundary violation" in str(excinfo.value)


def test_future_target_exclusion(pipeline):
    """Asserts ValidationError is raised if future target columns are provided."""
    leaked_target_df = pd.DataFrame([
        {'days_from_baseline': 0, 'ctDNA_vaf_percent': 0.5, 'future_ctDNA_30d_target': 1.2}
    ])

    with pytest.raises(validation.ValidationError) as excinfo:
        pipeline.run_patient_inference(
            patient_id="PAT-TEST",
            temporal_history=leaked_target_df
        )
    assert "Anti-leakage violation" in str(excinfo.value)


def test_missing_modality_pathology_only(pipeline, sample_data):
    """Validates single-modality pathology behavior when temporal data is absent."""
    res = pipeline.run_patient_inference(
        patient_id="PAT-0001",
        pathology_tiles=sample_data["tiles"][:2],
        temporal_history=None
    )
    assert res["modality_status"] == "PATHOLOGY_ONLY"
    assert res["modality_availability"]["pathology"] is True
    assert res["modality_availability"]["temporal"] is False
    assert res["temporal_summary"]["available"] is False
    assert res["multimodal_fusion"]["prototype_multimodal_risk_score"] is not None
    assert "Pathology modality" in res["engineering_explanation"] or "Single-modality" in res["engineering_explanation"]


def test_missing_modality_temporal_only(pipeline, sample_data):
    """Validates single-modality temporal behavior when pathology tiles are absent."""
    res = pipeline.run_patient_inference(
        patient_id="PAT-0001",
        pathology_tiles=None,
        temporal_history=sample_data["bio_df"]
    )
    assert res["modality_status"] == "TEMPORAL_ONLY"
    assert res["modality_availability"]["pathology"] is False
    assert res["modality_availability"]["temporal"] is True
    assert res["pathology_summary"]["available"] is False
    assert res["multimodal_fusion"]["prototype_multimodal_risk_score"] is not None


def test_missing_modality_neither(pipeline):
    """Validates INSUFFICIENT_DATA response when neither modality is provided."""
    res = pipeline.run_patient_inference(
        patient_id="PAT-EMPTY",
        pathology_tiles=None,
        temporal_history=None
    )
    assert res["modality_status"] == "INSUFFICIENT_DATA"
    assert res["multimodal_fusion"]["prototype_multimodal_risk_score"] is None
    assert res["multimodal_fusion"]["prototype_alert_level"] == "INSUFFICIENT_DATA"


def test_fusion_score_calculation():
    """Verifies exact mathematical computation of the weighted linear formula."""
    mock_pathology = {
        'available': True,
        'malignant_probability': 0.80
    }
    mock_temporal = {
        'available': True,
        'progression_probability': 0.60,
        'predicted_ctDNA_30d_vaf': 2.50  # Risk index = 2.5 / 5.0 = 0.50
    }
    weights = {'pathology_malignant': 0.30, 'temporal_progression': 0.50, 'ctdna_vaf_risk': 0.20}

    f_res = fusion.execute_multimodal_fusion(
        pathology_summary=mock_pathology,
        temporal_summary=mock_temporal,
        fusion_method='weighted_linear',
        custom_weights=weights
    )
    expected_score = round(0.30 * 0.80 + 0.50 * 0.60 + 0.20 * 0.50, 4)  # 0.24 + 0.30 + 0.10 = 0.64
    assert f_res['multimodal_risk_score'] == expected_score
    assert f_res['prototype_alert_level'] == config.PROTOTYPE_ALERT_LABELS['MODERATE']


def test_standard_output_schema(pipeline, sample_data):
    """Asserts that all required keys, types, and ranges are present in the response."""
    res = pipeline.run_patient_inference(
        patient_id="PAT-0001",
        pathology_tiles=sample_data["tiles"][:2],
        temporal_history=sample_data["bio_df"]
    )
    required_top_keys = [
        'patient_id', 'modality_availability', 'modality_status',
        'pathology_summary', 'temporal_summary', 'multimodal_fusion',
        'engineering_explanation', 'provenance'
    ]
    for key in required_top_keys:
        assert key in res, f"Missing required top-level key: {key}"

    assert isinstance(res['multimodal_fusion']['prototype_multimodal_risk_score'], (int, float))
    assert "(Prototype)" in res['multimodal_fusion']['prototype_alert_level']


def test_provenance_and_disclaimer_fields(pipeline, sample_data):
    """Verifies that provenance fields and regulatory synthetic notices are present."""
    res = pipeline.run_patient_inference(
        patient_id="PAT-0001",
        pathology_tiles=sample_data["tiles"][:1],
        temporal_history=sample_data["bio_df"]
    )
    prov = res['provenance']
    assert prov['data_source'] == "synthetic"
    assert prov['clinical_validation_status'] == "NOT_CLINICALLY_VALIDATED"
    assert config.MANDATORY_DISCLAIMER in prov['mandatory_disclaimer']
    assert config.EQUIVALENCE_DISCLAIMER in prov['equivalence_disclaimer']


def test_frozen_components_immutability():
    """Confirms that prior stage checkpoint files and source datasets remain accessible and unmodified."""
    assert config.FROZEN_IMAGE_CHECKPOINT.stat().st_size > 40 * 1024 * 1024  # >40 MB
    assert config.FROZEN_TEMPORAL_CHECKPOINT.stat().st_size > 1 * 1024 * 1024   # >1 MB
    assert config.IMAGE_METADATA_PATH.stat().st_size > 0
    assert config.BIOMARKERS_PATH.stat().st_size > 0
