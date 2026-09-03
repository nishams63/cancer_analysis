"""
Unit tests for the independent evaluation pipeline and locked-test artifacts.
Role: Evaluation Engineer.
"""

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

from data_loader import load_master_dataset, prepare_features_and_target, TARGET_MAPPING, REVERSE_TARGET_MAPPING
from feature_engineering import FeatureEngineer
from preprocessing import PreprocessingArtifactManager
from utils import patient_level_split, ThresholdAdjustedClassifier

MASTER_DATASET_PATH = os.path.abspath(os.path.join(EVAL_DIR, "..", "data-engineering", "data", "processed", "master_patient_dataset.csv"))
MODEL_PATH = os.path.abspath(os.path.join(EVAL_DIR, "..", "ml", "models", "best_model", "model.joblib"))
PREPROCESSOR_PATH = os.path.abspath(os.path.join(EVAL_DIR, "..", "ml", "artifacts", "preprocessor", "preprocessor.joblib"))
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
FIGURES_DIR = os.path.join(EVAL_DIR, "figures")


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
    assert r_class[2] == pytest.approx(0.5)


def test_final_metrics_json_structure():
    """Verify final_metrics.json artifact exists and contains all required sections."""
    final_metrics_path = os.path.join(RESULTS_DIR, "final_metrics.json")
    assert os.path.exists(final_metrics_path), "final_metrics.json must exist"
    
    with open(final_metrics_path, "r") as f:
        data = json.load(f)
        
    assert "overall_metrics" in data
    assert "per_class_metrics" in data
    assert "bootstrap_95_ci" in data
    assert "error_transitions" in data
    assert "high_risk_breakdown" in data
    assert "generalization_gap" in data
    assert "benchmark_comparison" in data
    
    om = data["overall_metrics"]
    assert om["macro_f1"] > 0.50
    assert om["high_risk_recall"] > 0.55
    assert om["accuracy"] > 0.55
    
    # Verify bootstrap CI bounds
    ci = data["bootstrap_95_ci"]
    for m in ["macro_f1", "high_risk_recall", "accuracy"]:
        assert ci[m]["ci_95_lower"] <= ci[m]["mean"] <= ci[m]["ci_95_upper"]


def test_classification_and_confusion_matrix_reports():
    """Verify classification report and confusion matrix CSV format."""
    cm_csv_path = os.path.join(RESULTS_DIR, "confusion_matrix.csv")
    assert os.path.exists(cm_csv_path), "confusion_matrix.csv must exist"
    
    df_cm = pd.read_csv(cm_csv_path, index_col=0)
    assert df_cm.shape == (3, 3)
    assert df_cm.values.sum() == 1750  # Exactly all test records


def test_predictions_artifact_alignment():
    """Verify test predictions artifact alignment with test set records."""
    preds_csv_path = os.path.join(RESULTS_DIR, "predictions.csv")
    assert os.path.exists(preds_csv_path), "predictions.csv must exist"
    
    df_preds = pd.read_csv(preds_csv_path)
    assert len(df_preds) == 1750
    assert set(df_preds["actual_label"].unique()) == {"Low", "Moderate", "High"}
    assert set(df_preds["predicted_label"].unique()) == {"Low", "Moderate", "High"}
    
    # Check probability normalization
    prob_sums = df_preds["prob_low"] + df_preds["prob_moderate"] + df_preds["prob_high"]
    assert np.allclose(prob_sums, 1.0, atol=1e-3)


def test_subgroup_metrics_artifact():
    """Verify subgroup metrics CSV exists and covers clinical categories."""
    sub_path = os.path.join(RESULTS_DIR, "subgroup_metrics.csv")
    assert os.path.exists(sub_path), "subgroup_metrics.csv must exist"
    
    df_sub = pd.read_csv(sub_path)
    assert len(df_sub) >= 10
    categories = set(df_sub["subgroup_category"].unique())
    assert "Sex" in categories
    assert "Cancer Type" in categories
    assert "Age Group" in categories


def test_figures_generation():
    """Verify generated evaluation figures exist and are non-empty."""
    cm_png = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    dist_png = os.path.join(FIGURES_DIR, "class_distribution.png")
    
    assert os.path.exists(cm_png), "figures/confusion_matrix.png must exist"
    assert os.path.getsize(cm_png) > 0
    assert os.path.exists(dist_png), "figures/class_distribution.png must exist"
    assert os.path.getsize(dist_png) > 0


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
    assert len(pm.feature_names_) == 113


def test_prediction_shape_with_v4_features():
    """Verify test prediction and probability matrix dimensions using V4 feature set."""
    raw_df = pd.read_csv(MASTER_DATASET_PATH)
    train_df, test_df = patient_level_split(raw_df, group_col="patient_id", target_col="toxicity_risk", test_size=0.20, random_state=42)
    
    X_test_raw, _, _ = prepare_features_and_target(test_df)
    fe = FeatureEngineer(include_engineered=True, include_expanded=True)
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
