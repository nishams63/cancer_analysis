"""
Unit tests for metric computation functions in evaluation module.
"""

import numpy as np
import pandas as pd
from classification_metrics import compute_classification_metrics
from calibration import compute_calibration_metrics
from extraction_metrics import evaluate_dataset_extractions


def test_classification_metrics_perfect():
    y_true = np.array(["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    y_pred = np.array(["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    classes = ["CRITICAL", "HIGH", "LOW", "MEDIUM"]

    metrics = compute_classification_metrics(y_true, y_pred, classes, "test")
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["critical_recall"] == 1.0


def test_classification_metrics_imperfect():
    y_true = np.array(["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    y_pred = np.array(["LOW", "MEDIUM", "HIGH", "HIGH"])  # Missed critical
    classes = ["CRITICAL", "HIGH", "LOW", "MEDIUM"]

    metrics = compute_classification_metrics(y_true, y_pred, classes, "test")
    assert metrics["accuracy"] == 0.75
    assert metrics["critical_recall"] == 0.0


def test_calibration_metrics_bounds():
    y_true_indices = np.array([0, 1, 2, 3])
    probs = np.array([
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.6, 0.2, 0.1],
        [0.05, 0.05, 0.8, 0.1],
        [0.1, 0.1, 0.1, 0.7]
    ])

    calib = compute_calibration_metrics(y_true_indices, probs, n_bins=5, task_name="test_calib")
    assert calib["probability_bounds_valid"] is True
    assert calib["probability_sum_to_one"] is True
    assert 0.0 <= calib["brier_score"] <= 2.0
    assert 0.0 <= calib["expected_calibration_error"] <= 1.0


def test_extraction_metrics_toy():
    dummy_df = pd.DataFrame([{
        "text": "Patient treated with Cisplatin 75 mg/m2.",
        "ner_entities": '[{"start": 21, "end": 30, "label": "DRUG_NAME", "text": "Cisplatin"}, {"start": 31, "end": 39, "label": "DOSAGE", "text": "75 mg/m2"}]'
    }])

    res_exact = evaluate_dataset_extractions(dummy_df, match_type="exact")
    res_relaxed = evaluate_dataset_extractions(dummy_df, match_type="relaxed")

    assert res_exact["evaluated_documents"] == 1
    assert 0.0 <= res_exact["macro_mean_f1"] <= 1.0
    assert 0.0 <= res_relaxed["macro_mean_f1"] <= 1.0
