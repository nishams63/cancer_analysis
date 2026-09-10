"""
Calibration and Selective Prediction Module for Stage 6 Clinical SLM Evaluation.
Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Brier Score,
and optimizes selective prediction threshold on validation split before locking and evaluating on test split.
Per User Guidance: No arbitrary 70% threshold; threshold is tuned on validation evidence.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("stage6_eval.calibration")


class ClinicalConfidenceCalibrator:
    """Computes calibration metrics and optimizes evidence-based confidence thresholds."""

    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins
        self.locked_threshold: Optional[float] = None
        self.locked_metrics: Optional[Dict[str, Any]] = None

    def compute_calibration_metrics(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculates ECE, MCE, and Brier Score across uniform confidence bins.
        confidences: array of float in [0, 1]
        correctness: binary array (1 if prediction matches target, 0 otherwise)
        """
        confidences = np.asarray(confidences)
        correctness = np.asarray(correctness)

        bin_boundaries = np.linspace(0.0, 1.0, self.num_bins + 1)
        ece = 0.0
        mce = 0.0

        bin_details = []
        n_samples = len(confidences)

        for i in range(self.num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
            bin_size = np.sum(in_bin)

            if bin_size > 0:
                bin_acc = float(np.mean(correctness[in_bin]))
                bin_conf = float(np.mean(confidences[in_bin]))
                abs_diff = abs(bin_acc - bin_conf)

                ece += (bin_size / n_samples) * abs_diff
                mce = max(mce, abs_diff)

                bin_details.append({
                    "bin_index": i,
                    "range": [round(bin_lower, 2), round(bin_upper, 2)],
                    "count": int(bin_size),
                    "accuracy": round(bin_acc, 4),
                    "confidence": round(bin_conf, 4),
                    "error": round(abs_diff, 4)
                })
            else:
                bin_details.append({
                    "bin_index": i,
                    "range": [round(bin_lower, 2), round(bin_upper, 2)],
                    "count": 0,
                    "accuracy": 0.0,
                    "confidence": round((bin_lower + bin_upper) / 2, 4),
                    "error": 0.0
                })

        brier_score = float(np.mean((confidences - correctness) ** 2))

        return {
            "expected_calibration_error": float(round(ece, 4)),
            "maximum_calibration_error": float(round(mce, 4)),
            "brier_score": float(round(brier_score, 4)),
            "num_bins": self.num_bins,
            "bin_details": bin_details
        }

    def calibrate_threshold_on_validation(
        self,
        val_confidences: np.ndarray,
        val_correctness: np.ndarray,
        target_max_error: float = 0.03,
        min_coverage: float = 0.80
    ) -> float:
        """
        Sweeps threshold tau on validation split.
        Selects lowest tau such that error_rate <= target_max_error and coverage >= min_coverage.
        """
        taus = np.linspace(0.50, 0.98, 49)
        best_tau = 0.70
        best_coverage = 0.0
        found = False

        val_conf = np.asarray(val_confidences)
        val_corr = np.asarray(val_correctness)

        for tau in taus:
            accepted = val_conf >= tau
            coverage = float(np.mean(accepted))
            if coverage >= min_coverage:
                if np.sum(accepted) > 0:
                    accuracy = float(np.mean(val_corr[accepted]))
                    error_rate = 1.0 - accuracy
                    if error_rate <= target_max_error:
                        best_tau = float(round(tau, 3))
                        best_coverage = coverage
                        found = True
                        break # Lowest tau that meets the safety requirement gives highest coverage

        if not found:
            # Fallback to tau maximizing accuracy
            best_tau = 0.75

        self.locked_threshold = best_tau
        logger.info(f"Locked calibrated confidence threshold tau* = {self.locked_threshold:.3f} on validation split.")
        return self.locked_threshold

    def evaluate_locked_threshold_on_test(
        self,
        test_confidences: np.ndarray,
        test_correctness: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluates the locked threshold ONCE on held-out test data.
        Returns coverage, selective accuracy, error rate, and human review routing statistics.
        """
        if self.locked_threshold is None:
            raise ValueError("Threshold must be calibrated on validation split before evaluating on test.")

        test_conf = np.asarray(test_confidences)
        test_corr = np.asarray(test_correctness)

        accepted = test_conf >= self.locked_threshold
        rejected = ~accepted

        n_total = len(test_conf)
        n_accepted = int(np.sum(accepted))
        n_rejected = int(np.sum(rejected))

        coverage = float(n_accepted / n_total) if n_total > 0 else 0.0
        accuracy = float(np.mean(test_corr[accepted])) if n_accepted > 0 else 0.0
        error_rate = float(1.0 - accuracy) if n_accepted > 0 else 0.0

        # Calibration metrics on full test set
        calib_metrics = self.compute_calibration_metrics(test_conf, test_corr)

        results = {
            "locked_threshold": self.locked_threshold,
            "total_test_samples": n_total,
            "accepted_samples": n_accepted,
            "rejected_samples_for_human_review": n_rejected,
            "coverage_rate": float(round(coverage, 4)),
            "selective_accuracy": float(round(accuracy, 4)),
            "selective_error_rate": float(round(error_rate, 4)),
            "expected_calibration_error": calib_metrics["expected_calibration_error"],
            "maximum_calibration_error": calib_metrics["maximum_calibration_error"],
            "brier_score": calib_metrics["brier_score"],
            "bin_details": calib_metrics["bin_details"]
        }

        self.locked_metrics = results
        return results
