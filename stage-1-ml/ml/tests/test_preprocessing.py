"""
Unit tests for data validation, feature engineering, and preprocessing.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_loader import (
    load_master_dataset,
    validate_dataset,
    prepare_features_and_target,
    TARGET_MAPPING,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)
from feature_engineering import FeatureEngineer
from preprocessing import PreprocessingArtifactManager


def test_master_dataset_loading_and_validation():
    """Verify master dataset loading, dimensions, and zero missing values."""
    df = load_master_dataset()
    val_res = validate_dataset(df)
    assert val_res["status"] == "PASSED"
    assert val_res["rows"] == 8754
    assert val_res["columns"] == 35


def test_prepare_features_and_target():
    """Verify feature/target separation and feature exclusions."""
    df = load_master_dataset()
    X, y, meta = prepare_features_and_target(df)
    
    assert X.shape[0] == 8754
    assert X.shape[1] == 30  # 20 numerical + 10 categorical
    assert len(y) == 8754
    assert set(y.unique()) == {0, 1, 2}
    
    # Excluded columns check
    assert "patient_id" not in X.columns
    assert "encounter_id" not in X.columns
    assert "observation_date" not in X.columns
    assert "treatment_response" not in X.columns
    assert "toxicity_risk" not in X.columns


def test_feature_engineering_transformation():
    """Verify feature engineer constructs expected domain features."""
    df = load_master_dataset()
    X, _, _ = prepare_features_and_target(df)
    
    fe = FeatureEngineer(include_engineered=True)
    X_fe = fe.transform(X)
    
    expected_engineered = [
        "blood_pressure_ratio", "pulse_pressure", "hematologic_risk_flag",
        "prior_toxicity_risk_flag", "comorbidity_age_interaction", "tumor_biomarker_index"
    ]
    for col in expected_engineered:
        assert col in X_fe.columns
        assert not X_fe[col].isnull().any()


def test_preprocessing_fit_transform():
    """Verify preprocessing pipeline fits and transforms cleanly."""
    df = load_master_dataset()
    X, _, _ = prepare_features_and_target(df)
    
    fe = FeatureEngineer(include_engineered=True)
    X_fe = fe.transform(X)
    
    num_cols = list(NUMERICAL_FEATURES) + [
        "blood_pressure_ratio", "pulse_pressure", "comorbidity_age_interaction",
        "tumor_biomarker_index", "hematologic_risk_flag", "prior_toxicity_risk_flag"
    ]
    cat_cols = list(CATEGORICAL_FEATURES)
    
    pm = PreprocessingArtifactManager()
    X_proc = pm.fit_transform(X_fe, num_cols, cat_cols)
    
    assert isinstance(X_proc, np.ndarray)
    assert X_proc.shape[0] == 8754
    assert X_proc.shape[1] > len(num_cols)  # due to OneHotEncoding
