"""
Stage 2 Deep Learning (DL) - Utility Functions
"""
import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, roc_auc_score
)
from typing import Dict, Any, Tuple, List

try:
    from . import config
except (ImportError, ValueError):
    import config

def set_seed(seed: int = config.RANDOM_SEED):
    """Sets random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, Any]:
    """Computes comprehensive multiclass classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_r = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_p = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_r = recall_score(y_true, y_pred, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        'accuracy': float(acc),
        'macro_precision': float(macro_p),
        'macro_recall': float(macro_r),
        'macro_f1': float(macro_f1),
        'per_class': {
            config.IDX_TO_CLASS[i]: {
                'precision': float(per_class_p[i]),
                'recall': float(per_class_r[i]),
                'f1': float(per_class_f1[i])
            }
            for i in range(config.NUM_CLASSES)
        },
        'confusion_matrix': cm.tolist()
    }
    return metrics

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes continuous forecasting metrics (MAE, RMSE, R²)."""
    # Filter NaNs if any
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0}

    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp)) if len(yt) > 1 else 0.0

    return {'mae': round(mae, 4), 'rmse': round(rmse, 4), 'r2': round(r2, 4)}

def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Computes binary classification metrics for recurrence/progression."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    yt, yp = y_true[mask], y_prob[mask]
    if len(yt) == 0:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'roc_auc': 0.0}

    y_pred = (yp >= threshold).astype(int)
    acc = float(accuracy_score(yt, y_pred))
    p = float(precision_score(yt, y_pred, zero_division=0))
    r = float(recall_score(yt, y_pred, zero_division=0))
    f1 = float(f1_score(yt, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(yt, yp))
    except Exception:
        auc = 0.5

    return {
        'accuracy': round(acc, 4),
        'precision': round(p, 4),
        'recall': round(r, 4),
        'f1': round(f1, 4),
        'roc_auc': round(auc, 4)
    }

def save_checkpoint(state: Dict[str, Any], filepath: str):
    """Saves model checkpoint with full reproducibility metadata."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)

def load_checkpoint(filepath: str, map_location: str = 'cpu') -> Dict[str, Any]:
    """Loads model checkpoint safely."""
    return torch.load(filepath, map_location=map_location, weights_only=False)

def plot_image_training_curves(history: Dict[str, List[float]], out_path: str):
    """Plots training and validation loss and macro F1 curves for CNN."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'o-', label='Train Loss', color='#1f77b4')
    ax1.plot(epochs, history['val_loss'], 's-', label='Val Loss', color='#d62728')
    ax1.set_title('Pathology CNN: Loss Curve', weight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epochs, history['train_f1'], 'o-', label='Train Macro F1', color='#2ca02c')
    ax2.plot(epochs, history['val_f1'], 's-', label='Val Macro F1', color='#ff7f0e')
    ax2.set_title('Pathology CNN: Macro F1 Curve', weight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro F1 Score')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle('Pathology Tile CNN Training & Validation Curves', weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_temporal_training_curves(history: Dict[str, List[float]], out_path: str):
    """Plots multi-task training and validation metrics for BiLSTM."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'o-', label='Train Total Loss', color='#1f77b4')
    ax1.plot(epochs, history['val_loss'], 's-', label='Val Total Loss', color='#d62728')
    ax1.set_title('Temporal BiLSTM: Multi-Task Loss', weight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Composite Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epochs, history['val_mae'], '^-', label='Val ctDNA MAE (%)', color='#9467bd')
    ax2.plot(epochs, history['val_f1'], 's-', label='Val Progression F1', color='#ff7f0e')
    ax2.set_title('Temporal BiLSTM: Validation Target Metrics', weight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Metric Value')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle('Longitudinal BiLSTM Training & Validation Curves', weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_confusion_matrix_heatmap(cm: np.ndarray, class_names: List[str], out_path: str):
    """Plots normalized confusion matrix heatmap."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=[c.capitalize() for c in class_names],
                yticklabels=[c.capitalize() for c in class_names], ax=ax)
    ax.set_title('Validation Split Confusion Matrix (Normalized)', weight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
