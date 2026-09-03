"""
Unit and Integration Tests for Class Balance and Patient Leakage Prevention.
Stage 1 ML: Oncology Toxicity Risk Multiclass Classification.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "src")
sys.path.insert(0, SRC_DIR)

from data_loader import (
    load_master_dataset,
    prepare_features_and_target,
    TARGET_MAPPING,
    REVERSE_TARGET_MAPPING,
    EXCLUDED_FEATURES_MAP
)
from feature_engineering import FeatureEngineer
from preprocessing import PreprocessingArtifactManager
from utils import patient_level_split
from experiment_class_balance import compute_fold_balanced_weights


@pytest.fixture(scope="module")
def dataset_split():
    raw_df = load_master_dataset()
    train_df, test_df = patient_level_split(
        raw_df, group_col="patient_id", target_col="toxicity_risk",
        test_size=0.20, random_state=42
    )
    return train_df, test_df


def test_locked_test_set_untouched_and_unbalanced(dataset_split):
    """Verify locked test set is completely separate and was never resampled or altered."""
    train_df, test_df = dataset_split
    assert len(test_df) == 1750, f"Expected 1750 test encounters, got {len(test_df)}"
    assert test_df["patient_id"].nunique() == 1200, f"Expected 1200 unique test patients"
    
    # Class distribution in locked test set must match the natural ground truth
    counts = test_df["toxicity_risk"].value_counts().to_dict()
    assert counts["Low"] == 942
    assert counts["Moderate"] == 474
    assert counts["High"] == 334


def test_patient_groups_strictly_separated(dataset_split):
    """Verify zero patient overlap between train and test sets."""
    train_df, test_df = dataset_split
    train_pids = set(train_df["patient_id"])
    test_pids = set(test_df["patient_id"])
    overlap = train_pids.intersection(test_pids)
    assert len(overlap) == 0, f"Patient leakage detected: {len(overlap)} overlapping patients!"


def test_cv_folds_patient_disjoint(dataset_split):
    """Verify all 5 CV folds have strictly disjoint patient sets."""
    train_df, _ = dataset_split
    X_raw, y, meta = prepare_features_and_target(train_df)
    patient_ids = meta["patient_id"].values
    
    sgkf = StratifiedGroupKFold(n_splits=5)
    for fold_i, (trn_idx, val_idx) in enumerate(sgkf.split(X_raw, y.values, groups=patient_ids)):
        trn_pts = set(patient_ids[trn_idx])
        val_pts = set(patient_ids[val_idx])
        overlap = trn_pts.intersection(val_pts)
        assert len(overlap) == 0, f"Fold {fold_i} has {len(overlap)} overlapping patients!"


def test_fold_local_weight_computation():
    """Verify that class weights are derived strictly from fold-local empirical distributions."""
    # Synthetic fold with known distribution: 50 Low (0), 30 Moderate (1), 20 High (2)
    y_synthetic = np.array([0]*50 + [1]*30 + [2]*20)
    w_default = compute_fold_balanced_weights(y_synthetic, mod_multiplier=1.0)
    
    # Formula: N / (3 * N_c)
    assert pytest.approx(w_default[0], rel=1e-3) == 100.0 / (3 * 50)
    assert pytest.approx(w_default[1], rel=1e-3) == 100.0 / (3 * 30)
    assert pytest.approx(w_default[2], rel=1e-3) == 100.0 / (3 * 20)
    
    # With moderate multiplier 1.4
    w_mod = compute_fold_balanced_weights(y_synthetic, mod_multiplier=1.4)
    assert pytest.approx(w_mod[1], rel=1e-3) == (100.0 / (3 * 30)) * 1.4
    assert pytest.approx(w_mod[0], rel=1e-3) == w_default[0]
    assert pytest.approx(w_mod[2], rel=1e-3) == w_default[2]


def test_feature_exclusion_leakage_free(dataset_split):
    """Verify that no forbidden columns leak into training features."""
    train_df, _ = dataset_split
    X_raw, _, _ = prepare_features_and_target(train_df)
    
    forbidden = list(EXCLUDED_FEATURES_MAP.keys())
    for col in forbidden:
        assert col not in X_raw.columns, f"Forbidden column '{col}' leaked into feature matrix!"


def test_prediction_output_classes_and_mapping():
    """Verify target mappings cover exactly Low, Moderate, and High."""
    assert TARGET_MAPPING == {"Low": 0, "Moderate": 1, "High": 2}
    assert REVERSE_TARGET_MAPPING == {0: "Low", 1: "Moderate", 2: "High"}


def test_v4_model_artifacts_intact():
    """Verify that existing Candidate V4 frozen model and preprocessor are preserved."""
    import joblib
    v4_model_path = os.path.join(os.path.dirname(CURRENT_DIR), "models", "best_model", "model.joblib")
    v4_preproc_path = os.path.join(os.path.dirname(CURRENT_DIR), "artifacts", "preprocessor", "preprocessor.joblib")
    
    assert os.path.exists(v4_model_path), "V4 model artifact missing!"
    assert os.path.exists(v4_preproc_path), "V4 preprocessor artifact missing!"
    
    model = joblib.load(v4_model_path)
    preproc = joblib.load(v4_preproc_path)
    assert hasattr(model, "predict_proba"), "Loaded V4 model missing predict_proba!"
    assert hasattr(preproc, "transform"), "Loaded V4 preprocessor missing transform!"
