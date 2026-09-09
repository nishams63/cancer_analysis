"""
Probability Calibration and Confidence Analysis Module for Stage 3 Clinical NLP.
Calculates Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
multi-class Brier score, probability bounds validity, and reliability diagram bins.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


def compute_calibration_metrics(
    y_true_indices: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
    task_name: str = "urgency"
) -> Dict[str, Any]:
    """
    Compute multi-class calibration metrics including ECE, MCE, and Brier score.
    Args:
        y_true_indices: integer array of true class indices (0 to K-1)
        probs: 2D array (N x K) of predicted class probabilities
        n_bins: number of equal-width confidence bins
        task_name: descriptive task name
    """
    n_samples, n_classes = probs.shape

    # 1. Probability validity checks
    min_prob = float(np.min(probs))
    max_prob = float(np.max(probs))
    row_sums = np.sum(probs, axis=1)
    sum_valid = bool(np.allclose(row_sums, 1.0, atol=1e-4))
    range_valid = bool(min_prob >= -1e-5 and max_prob <= 1.0 + 1e-5)

    # 2. Multi-class Brier Score
    one_hot = np.zeros_like(probs)
    for i, true_idx in enumerate(y_true_indices):
        one_hot[i, true_idx] = 1.0
    brier_score = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    # 3. Top-class Confidence and Accuracy
    top_preds = np.argmax(probs, axis=1)
    top_confs = np.max(probs, axis=1)
    accuracies = (top_preds == y_true_indices).astype(float)

    # 4. Binning for ECE and MCE
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_details = []
    ece = 0.0
    mce = 0.0

    for m in range(n_bins):
        bin_lower = bins[m]
        bin_upper = bins[m + 1]
        if m == n_bins - 1:
            in_bin = (top_confs >= bin_lower) & (top_confs <= bin_upper)
        else:
            in_bin = (top_confs >= bin_lower) & (top_confs < bin_upper)

        bin_size = int(np.sum(in_bin))
        if bin_size > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(top_confs[in_bin]))
            bin_err = abs(bin_acc - bin_conf)
            ece += (bin_size / n_samples) * bin_err
            mce = max(mce, bin_err)

            bin_details.append({
                "bin_index": m,
                "bin_lower": round(float(bin_lower), 2),
                "bin_upper": round(float(bin_upper), 2),
                "count": bin_size,
                "frequency": round(bin_size / n_samples, 4),
                "mean_confidence": round(bin_conf, 4),
                "mean_accuracy": round(bin_acc, 4),
                "calibration_gap": round(bin_err, 4)
            })
        else:
            bin_details.append({
                "bin_index": m,
                "bin_lower": round(float(bin_lower), 2),
                "bin_upper": round(float(bin_upper), 2),
                "count": 0,
                "frequency": 0.0,
                "mean_confidence": round((bin_lower + bin_upper) / 2.0, 4),
                "mean_accuracy": 0.0,
                "calibration_gap": 0.0
            })

    # Summary confidence statistics
    conf_stats = {
        "mean_confidence": round(float(np.mean(top_confs)), 4),
        "median_confidence": round(float(np.median(top_confs)), 4),
        "std_confidence": round(float(np.std(top_confs)), 4),
        "min_confidence": round(float(np.min(top_confs)), 4),
        "max_confidence": round(float(np.max(top_confs)), 4)
    }

    return {
        "task_name": task_name,
        "n_samples": n_samples,
        "n_classes": n_classes,
        "brier_score": round(brier_score, 4),
        "expected_calibration_error": round(float(ece), 4),
        "maximum_calibration_error": round(float(mce), 4),
        "probability_bounds_valid": range_valid,
        "probability_sum_to_one": sum_valid,
        "confidence_statistics": conf_stats,
        "bin_details": bin_details
    }
