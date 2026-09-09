"""
Bootstrap Confidence Intervals Module for Stage 3 Clinical NLP Evaluation.
Implements patient-clustered non-parametric bootstrap resampling to compute
valid 95% empirical confidence intervals for Accuracy, Macro F1, Macro Recall,
and Critical Class Recall while accounting for within-patient correlation.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score


def compute_patient_bootstrap_ci(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
    critical_class: Optional[str] = "CRITICAL"
) -> Dict[str, Any]:
    """
    Compute 95% bootstrap confidence intervals by resampling at the patient level.
    Resamples patient clusters with replacement to preserve within-patient narrative correlation.
    """
    rng = np.random.default_rng(random_seed)

    patient_ids = df["patient_id"].values
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)

    # Pre-map patient_id to array indices
    patient_to_indices = {}
    for idx, pid in enumerate(patient_ids):
        if pid not in patient_to_indices:
            patient_to_indices[pid] = []
        patient_to_indices[pid].append(idx)

    # Storage for bootstrap replicates
    boot_acc = []
    boot_macro_f1 = []
    boot_macro_recall = []
    boot_critical_recall = []
    boot_weighted_f1 = []

    for _ in range(n_iterations):
        # Sample patients with replacement
        sampled_pts = rng.choice(unique_patients, size=n_patients, replace=True)

        # Collect all document indices
        sampled_indices = []
        for pid in sampled_pts:
            sampled_indices.extend(patient_to_indices[pid])

        sub_y_true = y_true[sampled_indices]
        sub_y_pred = y_pred[sampled_indices]

        # Calculate metrics on sample
        acc = accuracy_score(sub_y_true, sub_y_pred)
        mf1 = f1_score(sub_y_true, sub_y_pred, average="macro", zero_division=0)
        mrec = recall_score(sub_y_true, sub_y_pred, average="macro", zero_division=0)
        wf1 = f1_score(sub_y_true, sub_y_pred, average="weighted", zero_division=0)

        boot_acc.append(acc)
        boot_macro_f1.append(mf1)
        boot_macro_recall.append(mrec)
        boot_weighted_f1.append(wf1)

        if critical_class and critical_class in sub_y_true:
            crit_mask = (sub_y_true == critical_class)
            if np.sum(crit_mask) > 0:
                crit_rec = recall_score(
                    crit_mask, (sub_y_pred == critical_class),
                    pos_label=True, zero_division=0
                )
                boot_critical_recall.append(crit_rec)

    alpha = (1.0 - confidence_level) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    def _ci_dict(arr: List[float], name: str) -> Dict[str, float]:
        if not arr:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std_err": 0.0}
        return {
            "mean": round(float(np.mean(arr)), 4),
            "std_err": round(float(np.std(arr)), 4),
            "ci_lower": round(float(np.percentile(arr, lower_pct)), 4),
            "ci_upper": round(float(np.percentile(arr, upper_pct)), 4)
        }

    results = {
        "resampling_unit": "patient_cluster",
        "n_iterations": n_iterations,
        "confidence_level": confidence_level,
        "random_seed": random_seed,
        "unique_patients_sampled": n_patients,
        "metrics": {
            "accuracy": _ci_dict(boot_acc, "accuracy"),
            "macro_f1": _ci_dict(boot_macro_f1, "macro_f1"),
            "macro_recall": _ci_dict(boot_macro_recall, "macro_recall"),
            "weighted_f1": _ci_dict(boot_weighted_f1, "weighted_f1")
        }
    }

    if boot_critical_recall:
        results["metrics"]["critical_recall"] = _ci_dict(boot_critical_recall, "critical_recall")

    return results
