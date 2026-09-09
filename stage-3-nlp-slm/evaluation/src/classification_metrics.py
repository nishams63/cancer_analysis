"""
Classification Metrics Module for Stage 3 Clinical NLP Evaluation.
Calculates Accuracy, Macro F1, Weighted F1, Macro Precision, Macro Recall,
per-class metrics, confusion matrices, and critical class recalls.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    task_name: str = "classification"
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics for a given task.
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    report = classification_report(
        y_true, y_pred,
        labels=class_names,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_norm = np.zeros_like(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    nonzero_mask = row_sums.flatten() > 0
    cm_norm[nonzero_mask] = cm[nonzero_mask] / row_sums[nonzero_mask]

    per_class = {}
    for c in class_names:
        if c in report:
            per_class[c] = {
                "precision": round(float(report[c]["precision"]), 4),
                "recall": round(float(report[c]["recall"]), 4),
                "f1": round(float(report[c]["f1-score"]), 4),
                "support": int(report[c]["support"])
            }
        else:
            per_class[c] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}

    metrics = {
        "task_name": task_name,
        "total_samples": len(y_true),
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.round(4).tolist(),
        "class_names": class_names
    }

    # Add specific high-risk class recall indicators if present
    if "CRITICAL" in per_class:
        metrics["critical_recall"] = per_class["CRITICAL"]["recall"]
    if "HIGH" in per_class:
        metrics["high_recall"] = per_class["HIGH"]["recall"]

    return metrics


def confusion_matrix_to_dataframe(cm: List[List[int]], class_names: List[str]) -> pd.DataFrame:
    """Format confusion matrix as a labeled pandas DataFrame."""
    return pd.DataFrame(cm, index=[f"True_{c}" for c in class_names], columns=[f"Pred_{c}" for c in class_names])
