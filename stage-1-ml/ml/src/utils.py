"""
Utility functions for patient-level dataset splitting, model evaluation, and reporting.
"""

from typing import Tuple, Dict, Any, List, Optional
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

CLASS_NAMES = ["Low", "Moderate", "High"]


def patient_level_split(
    df: pd.DataFrame,
    group_col: str = "patient_id",
    target_col: str = "toxicity_risk",
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs patient-level train/test split ensuring zero overlap of patient_id between train and test sets.
    Stratifies unique patients based on their primary toxicity risk category.
    """
    # Map patient to their most frequent / primary toxicity risk for stratification
    patient_target_map = df.groupby(group_col)[target_col].agg(lambda x: x.mode()[0]).to_dict()
    unique_patients = np.array(list(patient_target_map.keys()))
    patient_targets = np.array([patient_target_map[pid] for pid in unique_patients])
    
    train_pids, test_pids = train_test_split(
        unique_patients,
        test_size=test_size,
        random_state=random_state,
        stratify=patient_targets
    )
    
    train_set_pids = set(train_pids)
    test_set_pids = set(test_pids)
    
    # Assert zero patient overlap
    overlap = train_set_pids.intersection(test_set_pids)
    if len(overlap) > 0:
        raise ValueError(f"CRITICAL ERROR: Patient overlap detected between train and test splits! Overlap count: {len(overlap)}")
        
    train_df = df[df[group_col].isin(train_set_pids)].copy().reset_index(drop=True)
    test_df = df[df[group_col].isin(test_set_pids)].copy().reset_index(drop=True)
    
    return train_df, test_df


def evaluate_multiclass_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: List[str] = CLASS_NAMES
) -> Dict[str, Any]:
    """
    Calculates comprehensive multiclass classification evaluation metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    
    # Macro metrics
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    # Weighted metrics
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    # Per-class metrics
    prec_per_class, rec_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    
    # High-risk (class index 2) recall
    high_risk_idx = 2
    high_risk_recall = float(rec_per_class[high_risk_idx]) if len(rec_per_class) > high_risk_idx else 0.0
    
    per_class_dict = {}
    for i, cname in enumerate(class_names):
        per_class_dict[cname] = {
            "precision": float(prec_per_class[i]),
            "recall": float(rec_per_class[i]),
            "f1": float(f1_per_class[i]),
            "support": int(support_per_class[i])
        }
        
    return {
        "accuracy": float(acc),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "high_risk_recall": float(high_risk_recall),
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist()
    }


def plot_confusion_matrix_figure(
    cm: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    title: str = "Confusion Matrix",
    output_path: Optional[str] = None
):
    """
    Renders and saves a formatted confusion matrix heatmap figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Predicted Toxicity Risk", fontsize=10)
    ax.set_ylabel("Actual Toxicity Risk", fontsize=10)
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_feature_importance_figure(
    feature_names: List[str],
    importance_scores: np.ndarray,
    top_n: int = 15,
    title: str = "Top Feature Importances",
    output_path: Optional[str] = None
):
    """
    Renders and saves a horizontal bar chart of feature importances.
    """
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importance_scores
    }).sort_values(by="importance", ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        data=df_imp,
        x="importance",
        y="feature",
        palette="viridis",
        ax=ax
    )
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Importance Score", fontsize=10)
    ax.set_ylabel("Feature", fontsize=10)
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close(fig)
