"""
Unit tests for Calibration and Evidence-Based Selective Prediction.
"""

import numpy as np
import pytest
from calibration import ClinicalConfidenceCalibrator


def test_calibration_metrics_computation():
    calibrator = ClinicalConfidenceCalibrator(num_bins=5)
    # Perfect calibration synthetic data: confidences match accuracies
    confs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    corrs = np.array([0, 0, 1, 1, 1])

    metrics = calibrator.compute_calibration_metrics(confs, corrs)
    assert "expected_calibration_error" in metrics
    assert "maximum_calibration_error" in metrics
    assert "brier_score" in metrics
    assert metrics["expected_calibration_error"] >= 0.0
    assert metrics["brier_score"] >= 0.0
    assert len(metrics["bin_details"]) == 5


def test_evidence_based_threshold_calibration_and_evaluation():
    calibrator = ClinicalConfidenceCalibrator(num_bins=10)

    # 100 validation samples
    np.random.seed(42)
    val_confs = np.concatenate([np.random.uniform(0.80, 0.98, 90), np.random.uniform(0.50, 0.70, 10)])
    val_corrs = np.concatenate([np.ones(90, dtype=int), np.zeros(10, dtype=int)])

    # Calibrate on validation
    tau = calibrator.calibrate_threshold_on_validation(val_confs, val_corrs, target_max_error=0.05)
    assert 0.50 <= tau <= 0.95
    assert calibrator.locked_threshold == tau

    # Evaluate on held-out test
    test_confs = np.concatenate([np.random.uniform(0.85, 0.99, 95), np.random.uniform(0.50, 0.65, 5)])
    test_corrs = np.concatenate([np.ones(95, dtype=int), np.zeros(5, dtype=int)])

    eval_res = calibrator.evaluate_locked_threshold_on_test(test_confs, test_corrs)
    assert eval_res["coverage_rate"] >= 0.85
    assert eval_res["selective_accuracy"] >= 0.95
    assert eval_res["selective_error_rate"] <= 0.05
    assert eval_res["rejected_samples_for_human_review"] >= 1
