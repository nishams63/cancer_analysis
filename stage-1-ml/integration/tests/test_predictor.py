"""
Unit tests for V4InferenceEngine and frozen artifact loading.
Role: Integration Engineer.

Covers:
- TEST 1: Model artifact loading
- TEST 2: Preprocessor artifact loading
- TEST 3: Target mapping loading
- TEST 9: Deterministic prediction output
- TEST 10: Feature preparation compatibility (41 features -> 113 preprocessed)
- TEST 11: Batch prediction functionality
- TEST 12: Useful error on missing artifact path
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Path configuration
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from predictor import V4InferenceEngine


@pytest.fixture
def sample_patient_record():
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


def test_1_model_artifact_loading():
    """TEST 1: Verify the V4 model artifact loads successfully and has prediction methods."""
    engine = V4InferenceEngine()
    assert engine.model is not None
    assert hasattr(engine.model, "predict")
    assert hasattr(engine.model, "predict_proba")


def test_2_preprocessor_artifact_loading():
    """TEST 2: Verify the V4 preprocessor loads successfully and contains expected feature names."""
    engine = V4InferenceEngine()
    assert engine.preprocessor is not None
    assert hasattr(engine.preprocessor, "transform")
    assert len(engine.preprocessor.feature_names_) == 113


def test_3_target_mapping_loading():
    """TEST 3: Verify the target mapping loads successfully with explicit class mappings."""
    engine = V4InferenceEngine()
    assert engine.target_mapping == {"Low": 0, "Moderate": 1, "High": 2}
    assert engine.reverse_mapping == {0: "Low", 1: "Moderate", 2: "High"}


def test_9_deterministic_prediction_output(sample_patient_record):
    """TEST 9: Run the same valid input multiple times and verify identical, deterministic output."""
    engine = V4InferenceEngine()
    res1 = engine.predict_single(sample_patient_record)
    res2 = engine.predict_single(sample_patient_record)
    res3 = engine.predict_single(sample_patient_record)

    assert res1["predicted_toxicity_risk"] == res2["predicted_toxicity_risk"] == res3["predicted_toxicity_risk"]
    assert res1["probabilities"] == res2["probabilities"] == res3["probabilities"]


def test_10_feature_preparation_compatibility(sample_patient_record):
    """TEST 10: Verify feature engineering produces all 41 features compatible with preprocessor."""
    engine = V4InferenceEngine()
    df_raw = pd.DataFrame([sample_patient_record])
    df_fe = engine.feature_engineer.transform(df_raw)

    # Check key V4 engineered features are present
    assert "cumulative_treatment_load" in df_fe.columns
    assert "organ_impairment_index" in df_fe.columns
    assert "vital_instability_score" in df_fe.columns
    assert "genomic_instability_score" in df_fe.columns
    assert "biomarker_severity_weight" in df_fe.columns
    assert df_fe.shape[1] == 41

    # Check transformed matrix shape matches training
    X_proc = engine.preprocessor.transform(df_fe)
    assert X_proc.shape == (1, 113)


def test_11_batch_prediction_multiple_inputs(sample_patient_record):
    """TEST 11: Verify batch prediction evaluates multiple inputs cleanly."""
    engine = V4InferenceEngine()
    rec2 = dict(sample_patient_record)
    rec2["age"] = 75.0
    rec2["creatinine_level"] = 2.8
    rec2["previous_toxicity_grade"] = 3.0

    batch = [sample_patient_record, rec2]
    results = engine.predict_batch(batch)

    assert len(results) == 2
    for r in results:
        assert r["predicted_toxicity_risk"] in ["Low", "Moderate", "High"]
        assert "probabilities" in r
        assert set(r["probabilities"].keys()) == {"Low", "Moderate", "High"}


def test_12_error_on_missing_artifact_path():
    """TEST 12: Verify a clear FileNotFoundError is raised if an artifact path is missing or invalid."""
    with pytest.raises(FileNotFoundError, match="V4 Model artifact not found"):
        V4InferenceEngine(model_path="non_existent_path_to_model.joblib")
