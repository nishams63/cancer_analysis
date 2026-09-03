import os
import sys
import pytest
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Ensure evaluation and ml/src paths are accessible
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
ML_SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "ml", "src"))
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)
if ML_SRC_DIR not in sys.path:
    sys.path.insert(0, ML_SRC_DIR)

from evaluation import validate_probabilities, run_evaluation, TARGET_MAPPING, REVERSE_TARGET_MAPPING
from data_loader import load_master_dataset, prepare_features_and_target
from feature_engineering import FeatureEngineer
from preprocessing import PreprocessingArtifactManager
from utils import patient_level_split

MASTER_DATASET_PATH = os.path.abspath(os.path.join(EVAL_DIR, "..", "data-engineering", "data", "processed", "master_patient_dataset.csv"))
MODEL_PATH = os.path.abspath(os.path.join(EVAL_DIR, "..", "ml", "models", "best_model", "model.joblib"))
PREPROCESSOR_PATH = os.path.abspath(os.path.join(EVAL_DIR, "..", "ml", "artifacts", "preprocessor", "preprocessor.joblib"))


def test_metric_calculation():
    """Verify standard metric calculations on a known dummy array."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 2])
    
    acc = accuracy_score(y_true, y_pred)
    assert acc == pytest.approx(5.0 / 6.0)
    
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    assert f1_macro > 0.0
    
    p_class, r_class, f1_class, s_class = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1, 2])
    assert len(f1_class) == 3
    assert r_class[2] == pytest.approx(0.5)  # High risk recall


def test_classification_report_generation():
    """Verify classification report output CSV format and values."""
    metrics_path = os.path.join(EVAL_DIR, "results", "metrics.json")
    report_csv_path = os.path.join(EVAL_DIR, "results", "classification_report.csv")
    
    assert os.path.exists(metrics_path), "metrics.json must exist"
    assert os.path.exists(report_csv_path), "classification_report.csv must exist"
    
    df_rep = pd.read_csv(report_csv_path)
    assert list(df_rep.columns) == ["class", "encoded_label", "precision", "recall", "f1_score", "support"]
    assert len(df_rep) == 3
    assert list(df_rep["class"]) == ["Low", "Moderate", "High"]


def test_confusion_matrix_generation():
    """Verify confusion matrix image artifact exists and is non-empty."""
    cm_png_path = os.path.join(EVAL_DIR, "results", "confusion_matrix.png")
    assert os.path.exists(cm_png_path), "confusion_matrix.png artifact must exist"
    assert os.path.getsize(cm_png_path) > 0


def test_probability_validation():
    """Test probability validator with valid probability matrix."""
    y_prob_valid = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.3, 0.5]
    ])
    y_pred_valid = np.array([0, 1, 2])
    
    assert validate_probabilities(y_prob_valid, y_pred_valid) is True


def test_invalid_nan_probability_detection():
    """Assert that probability validator raises ValueError on NaN probabilities."""
    y_prob_nan = np.array([
        [0.7, np.nan, 0.1],
        [0.1, 0.8, 0.1]
    ])
    y_pred_dummy = np.array([0, 1])
    
    with pytest.raises(ValueError, match="NaN values detected"):
        validate_probabilities(y_prob_nan, y_pred_dummy)


def test_target_mapping():
    """Verify target class mapping integrity."""
    assert TARGET_MAPPING["Low"] == 0
    assert TARGET_MAPPING["Moderate"] == 1
    assert TARGET_MAPPING["High"] == 2
    
    assert REVERSE_TARGET_MAPPING[0] == "Low"
    assert REVERSE_TARGET_MAPPING[1] == "Moderate"
    assert REVERSE_TARGET_MAPPING[2] == "High"


def test_model_artifact_loading():
    """Verify saved ML model artifact loads successfully."""
    assert os.path.exists(MODEL_PATH), "Saved model artifact must exist"
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict"), "Loaded model must have a predict method"
    assert hasattr(model, "predict_proba"), "Loaded model must have a predict_proba method"


def test_preprocessor_artifact_loading():
    """Verify saved preprocessor artifact loads successfully."""
    assert os.path.exists(PREPROCESSOR_PATH), "Saved preprocessor artifact must exist"
    pm = PreprocessingArtifactManager.load(PREPROCESSOR_PATH)
    assert hasattr(pm, "transform"), "Loaded preprocessor manager must have a transform method"


def test_prediction_shape():
    """Verify test prediction and probability matrix dimensions."""
    raw_df = pd.read_csv(MASTER_DATASET_PATH)
    train_df, test_df = patient_level_split(raw_df, group_col="patient_id", target_col="toxicity_risk", test_size=0.20, random_state=42)
    
    X_test_raw, _, _ = prepare_features_and_target(test_df)
    fe = FeatureEngineer(include_engineered=True)
    X_test_fe = fe.transform(X_test_raw)
    
    pm = PreprocessingArtifactManager.load(PREPROCESSOR_PATH)
    X_test_proc = pm.transform(X_test_fe)
    
    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test_proc)
    y_prob = model.predict_proba(X_test_proc)
    
    assert len(y_pred) == len(test_df)
    assert y_prob.shape == (len(test_df), 3)


def test_patient_overlap_detection():
    """Verify patient-level train/test split has strictly zero patient overlap."""
    raw_df = pd.read_csv(MASTER_DATASET_PATH)
    train_df, test_df = patient_level_split(raw_df, group_col="patient_id", target_col="toxicity_risk", test_size=0.20, random_state=42)
    
    train_patients = set(train_df["patient_id"])
    test_patients = set(test_df["patient_id"])
    overlap = train_patients.intersection(test_patients)
    
    assert len(overlap) == 0, f"Expected 0 patient overlap, found {len(overlap)}"
