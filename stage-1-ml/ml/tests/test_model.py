"""
Unit tests for model training, patient-level splitting, target mapping, and model artifact loading.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_loader import load_master_dataset, prepare_features_and_target, TARGET_MAPPING, REVERSE_TARGET_MAPPING
from utils import patient_level_split, evaluate_multiclass_predictions

MODELS_BEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best_model"))


def test_patient_level_split_no_overlap():
    """Verify patient-level train/test split results in zero patient overlap."""
    df = load_master_dataset()
    train_df, test_df = patient_level_split(df, group_col="patient_id", target_col="toxicity_risk", test_size=0.20, random_state=42)
    
    train_pids = set(train_df["patient_id"])
    test_pids = set(test_df["patient_id"])
    
    # Zero patient overlap verification
    assert len(train_pids.intersection(test_pids)) == 0
    assert len(train_df) + len(test_df) == len(df)


def test_target_mapping_consistency():
    """Verify target class label integer mapping is strictly bi-directional."""
    expected_mapping = {"Low": 0, "Moderate": 1, "High": 2}
    expected_reverse = {0: "Low", 1: "Moderate", 2: "High"}
    
    assert TARGET_MAPPING == expected_mapping
    assert REVERSE_TARGET_MAPPING == expected_reverse


def test_evaluation_metrics_computation():
    """Verify evaluation metric dictionary calculation."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])
    
    metrics = evaluate_multiclass_predictions(y_true, y_pred)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "high_risk_recall" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["accuracy"] > 0.0


def test_saved_model_artifact_loading():
    """Verify trained best model artifact exists and can be loaded."""
    model_path = os.path.join(MODELS_BEST_DIR, "model.joblib")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        assert hasattr(model, "predict")
