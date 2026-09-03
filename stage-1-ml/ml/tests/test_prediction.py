"""
Unit tests for single and batch toxicity risk inference module.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from predict import ToxicityRiskPredictor, predict_patient_toxicity

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model", "model.joblib")


@pytest.fixture
def sample_patient_dict():
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


def test_prediction_single_patient(sample_patient_dict):
    """Verify single patient inference returns expected JSON structure and valid class."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Model artifact not yet trained.")
        
    result = predict_patient_toxicity(sample_patient_dict)
    
    assert "predicted_risk" in result
    assert result["predicted_risk"] in ["Low", "Moderate", "High"]
    assert "probabilities" in result
    assert "Low" in result["probabilities"]
    assert "Moderate" in result["probabilities"]
    assert "High" in result["probabilities"]
    assert "disclaimer" in result
    
    # Verify probability normalization
    prob_sum = sum(result["probabilities"].values())
    assert pytest.approx(prob_sum, 1e-3) == 1.0


def test_prediction_batch(sample_patient_dict):
    """Verify batch patient inference on DataFrame input."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Model artifact not yet trained.")
        
    df_batch = pd.DataFrame([sample_patient_dict, sample_patient_dict])
    predictor = ToxicityRiskPredictor()
    results = predictor.predict_batch(df_batch)
    
    assert len(results) == 2
    for res in results:
        assert res["predicted_risk"] in ["Low", "Moderate", "High"]
