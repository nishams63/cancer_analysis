"""
Utility Functions and Visualization Helpers for Stage 3 Clinical NLP Evaluation.
Provides JSON serialization, markdown table generation, and publication-grade matplotlib plotting.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types and Path objects."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        return super().default(obj)


def save_json(data: Dict[str, Any], file_path: Path) -> Path:
    """Save dictionary to JSON with formatting and NumPy serialization."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=NumpyJSONEncoder)
    return file_path


def load_json(file_path: Path) -> Dict[str, Any]:
    """Load JSON file safely."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_confusion_matrix(
    cm_matrix: List[List[int]],
    class_names: List[str],
    title: str,
    output_path: Path,
    normalize: bool = True
) -> Path:
    """Plot publication-grade confusion matrix heatmap."""
    cm = np.array(cm_matrix)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        plot_data = cm / row_sums
        fmt = ".2f"
        cbar_label = "Normalized Proportion"
    else:
        plot_data = cm
        fmt = "d"
        cbar_label = "Document Count"

    plt.figure(figsize=(7, 5.5), dpi=300)
    sns.heatmap(
        plot_data,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": cbar_label},
        square=True
    )
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=10, labelpad=8)
    plt.ylabel("True Class", fontsize=10, labelpad=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_per_class_f1_bars(
    per_class_metrics: Dict[str, Dict[str, float]],
    title: str,
    output_path: Path
) -> Path:
    """Plot per-class Precision, Recall, and F1 bar chart."""
    classes = list(per_class_metrics.keys())
    p_vals = [per_class_metrics[c]["precision"] for c in classes]
    r_vals = [per_class_metrics[c]["recall"] for c in classes]
    f1_vals = [per_class_metrics[c]["f1"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(8, 4.8), dpi=300)
    plt.bar(x - width, p_vals, width, label="Precision", color="#3b82f6", alpha=0.9)
    plt.bar(x, r_vals, width, label="Recall", color="#10b981", alpha=0.9)
    plt.bar(x + width, f1_vals, width, label="F1-Score", color="#8b5cf6", alpha=0.9)

    plt.xlabel("Class", fontsize=10, fontweight="bold")
    plt.ylabel("Metric Score", fontsize=10, fontweight="bold")
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.xticks(x, classes, rotation=15, ha="right")
    plt.ylim(0.0, 1.05)
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_calibration_curve(
    bin_details: List[Dict[str, Any]],
    ece: float,
    title: str,
    output_path: Path
) -> Path:
    """Plot reliability diagram for probability calibration."""
    confs = [b["mean_confidence"] for b in bin_details if b["count"] > 0]
    accs = [b["mean_accuracy"] for b in bin_details if b["count"] > 0]

    plt.figure(figsize=(6, 5.5), dpi=300)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.plot(confs, accs, marker="o", color="#ef4444", linewidth=2, label=f"Model (ECE = {ece:.4f})")
    plt.bar([b["mean_confidence"] for b in bin_details],
            [b["mean_accuracy"] for b in bin_details],
            width=0.08, alpha=0.2, color="#3b82f6", edgecolor="none")

    plt.xlabel("Mean Predicted Confidence", fontsize=10, fontweight="bold")
    plt.ylabel("Observed Accuracy", fontsize=10, fontweight="bold")
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.legend(frameon=True, loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_generalization_comparison(
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    metric_names: List[str],
    output_path: Path,
    title: str = "Validation vs. Locked Test Performance"
) -> Path:
    """Plot side-by-side comparison of Validation vs Locked Test metrics."""
    x = np.arange(len(metric_names))
    width = 0.35

    val_vals = [val_metrics.get(m, 0.0) for m in metric_names]
    test_vals = [test_metrics.get(m, 0.0) for m in metric_names]

    plt.figure(figsize=(7, 4.5), dpi=300)
    plt.bar(x - width / 2, val_vals, width, label="Validation (N=909)", color="#3b82f6", alpha=0.9)
    plt.bar(x + width / 2, test_vals, width, label="Locked Test (N=928)", color="#f59e0b", alpha=0.9)

    plt.xlabel("Metric", fontsize=10, fontweight="bold")
    plt.ylabel("Score", fontsize=10, fontweight="bold")
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.xticks(x, [m.replace("_", " ").title() for m in metric_names])
    plt.ylim(0.0, 1.05)
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path
