"""
Stage 2 Deep Learning - Independent Longitudinal Sequence Evaluation on Locked Test Set
Evaluates frozen best_temporal_lstm.pt on 150 locked test patient trajectories.
Computes dual-head metrics:
  - Head A (ctDNA 30d Regression): MAE, RMSE, Pearson r, Spearman rho, R2, MAPE
  - Head B (Progression Trend Classification): Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from . import config
except (ImportError, ValueError):
    import config

# Import temporal model architecture from DL module
sys.path.insert(0, str(config.STAGE_2_DIR / 'dl'))
from src.temporal_model import build_temporal_model


def prepare_test_patient_sequences(bio_path: Path, norm_params: dict, features_list: list) -> list:
    """Prepares padded sequences dynamically for the 150 locked test patients."""
    df = pd.read_csv(bio_path)
    # Strict anti-leakage filter: only historical input window (<= 90d)
    test_df = df[(df['split'] == 'test') & (df['is_input_window'] == 1) & (df['days_from_baseline'] <= config.FORECAST_SPLIT_DAY)].copy()
    test_df = test_df.sort_values(['patient_id', 'days_from_baseline'])

    continuous_cols = [f for f in features_list if not f.endswith('_missing')]
    mask_cols = [f for f in features_list if f.endswith('_missing')]

    sequences = []
    for pat_id, grp in test_df.groupby('patient_id'):
        grp_c = grp.copy()
        # Impute forward fill then 0
        for col in continuous_cols:
            if col in grp_c.columns:
                grp_c[col] = grp_c[col].ffill().fillna(0.0)
            else:
                grp_c[col] = 0.0

        norm_matrix = np.zeros((len(grp_c), len(continuous_cols)), dtype=np.float32)
        for i, col in enumerate(continuous_cols):
            if col in norm_params:
                mean_v, std_v = norm_params[col]
            else:
                mean_v, std_v = 0.0, 1.0
            norm_matrix[:, i] = (grp_c[col].values - mean_v) / (std_v + 1e-6)

        mask_matrix = np.zeros((len(grp_c), len(mask_cols)), dtype=np.float32)
        for i, col in enumerate(mask_cols):
            if col in grp_c.columns:
                mask_matrix[:, i] = grp_c[col].values.astype(np.float32)

        feat_matrix = np.concatenate([norm_matrix, mask_matrix], axis=1)
        actual_len = len(feat_matrix)

        # Landmark target: last valid historical target for ctDNA (handles clinical missingness)
        valid_ctdna_series = grp_c[config.TARGET_REGRESSION].dropna()
        if len(valid_ctdna_series) > 0:
            target_ctdna = float(valid_ctdna_series.iloc[-1])
        else:
            target_ctdna = 0.0

        target_prog = int(grp_c.iloc[-1][config.TARGET_CLASSIFICATION])
        last_row_had_target = pd.notna(grp_c.iloc[-1][config.TARGET_REGRESSION])

        sequences.append({
            'patient_id': pat_id,
            'features': torch.tensor(feat_matrix, dtype=torch.float32),
            'length': actual_len,
            'target_ctdna': target_ctdna,
            'target_progression': target_prog,
            'last_row_had_target': last_row_had_target
        })

    return sequences


def evaluate_temporal_model(device: str = 'cpu') -> dict:
    print("\n" + "=" * 60)
    print("EVALUATING FROZEN TEMPORAL BiLSTM ON LOCKED TEST SET")
    print("=" * 60)

    chk_path = config.FROZEN_TEMPORAL_CHECKPOINT
    if not chk_path.exists():
        raise FileNotFoundError(f"Frozen checkpoint not found at: {chk_path}")

    checkpoint = torch.load(chk_path, map_location=device)
    arch = checkpoint.get('model_architecture', 'lstm')
    hidden_dim = checkpoint.get('hidden_dim', 64)
    norm_params = checkpoint.get('norm_params', {})
    features_list = checkpoint.get('temporal_features', config.get_checkpoint_temporal_features())

    print(f"Loaded checkpoint: {chk_path.name}")
    print(f"Architecture: {arch} | Hidden Dim: {hidden_dim} | Input Features: {len(features_list)}")

    model = build_temporal_model(arch=arch, input_dim=len(features_list), hidden_dim=hidden_dim)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    sequences = prepare_test_patient_sequences(config.BIOMARKERS_PATH, norm_params, features_list)
    print(f"Loaded test sequences for {len(sequences)} unique patients.")

    all_preds_ctdna = []
    all_targets_ctdna = []
    all_pred_logits = []
    all_targets_prog = []
    all_patient_ids = []
    strictly_last_row_mask = []

    with torch.no_grad():
        for seq_dict in sequences:
            # Batch of 1
            x = seq_dict['features'].unsqueeze(0).to(device)  # (1, L, D)
            lens = torch.tensor([seq_dict['length']], dtype=torch.long)

            pred_ctdna, pred_logits = model(x, lens)

            all_preds_ctdna.append(max(0.0, float(pred_ctdna.item())))
            all_targets_ctdna.append(seq_dict['target_ctdna'])
            all_pred_logits.append(float(pred_logits.item()))
            all_targets_prog.append(seq_dict['target_progression'])
            all_patient_ids.append(seq_dict['patient_id'])
            strictly_last_row_mask.append(seq_dict['last_row_had_target'])

    all_preds_ctdna = np.array(all_preds_ctdna)
    all_targets_ctdna = np.array(all_targets_ctdna)
    all_pred_logits = np.array(all_pred_logits)
    all_targets_prog = np.array(all_targets_prog)
    strictly_last_row_mask = np.array(strictly_last_row_mask)

    # Probabilities & classes
    all_pred_probs = 1.0 / (1.0 + np.exp(-all_pred_logits))
    all_pred_classes = (all_pred_probs >= 0.5).astype(int)

    # 1. Regression Metrics (Head A: ctDNA 30d VAF on all 150 patients)
    mae = mean_absolute_error(all_targets_ctdna, all_preds_ctdna)
    rmse = np.sqrt(mean_squared_error(all_targets_ctdna, all_preds_ctdna))
    r2 = r2_score(all_targets_ctdna, all_preds_ctdna)
    p_r, p_pval = pearsonr(all_targets_ctdna, all_preds_ctdna)
    s_rho, s_pval = spearmanr(all_targets_ctdna, all_preds_ctdna)
    mape = np.mean(np.abs((all_targets_ctdna - all_preds_ctdna) / np.clip(all_targets_ctdna, 1e-4, None))) * 100

    # Also compute on strictly last-row non-null subset (N=141)
    mae_141 = mean_absolute_error(all_targets_ctdna[strictly_last_row_mask], all_preds_ctdna[strictly_last_row_mask])
    rmse_141 = np.sqrt(mean_squared_error(all_targets_ctdna[strictly_last_row_mask], all_preds_ctdna[strictly_last_row_mask]))
    r2_141 = r2_score(all_targets_ctdna[strictly_last_row_mask], all_preds_ctdna[strictly_last_row_mask])

    print(f"\n--- Head A (ctDNA 30d Regression) Locked Test Metrics (N=150) ---")
    print(f"MAE:         {mae:.4f} (N=141 strictly last row: {mae_141:.4f})")
    print(f"RMSE:        {rmse:.4f} (N=141 strictly last row: {rmse_141:.4f})")
    print(f"R2 Score:    {r2:.4f} (N=141 strictly last row: {r2_141:.4f})")
    print(f"Pearson r:   {p_r:.4f} (p={p_pval:.2e})")
    print(f"Spearman rho:{s_rho:.4f} (p={s_pval:.2e})")
    print(f"MAPE:        {mape:.2f}%")

    # 2. Classification Metrics (Head B: Progression Trend on all 150 patients)
    acc = accuracy_score(all_targets_prog, all_pred_classes)
    bacc = balanced_accuracy_score(all_targets_prog, all_pred_classes)
    p, r, f1, _ = precision_recall_fscore_support(all_targets_prog, all_pred_classes, average='binary', zero_division=0)
    roc_auc = roc_auc_score(all_targets_prog, all_pred_probs)
    pr_auc = average_precision_score(all_targets_prog, all_pred_probs)
    cm = confusion_matrix(all_targets_prog, all_pred_classes, labels=[0, 1])

    print(f"\n--- Head B (Progression Trend Binary Classification) Locked Test Metrics ---")
    print(f"Accuracy:    {acc * 100:.2f}%")
    print(f"Balanced Acc:{bacc * 100:.2f}%")
    print(f"Precision:   {p:.4f}")
    print(f"Recall:      {r:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"PR-AUC:      {pr_auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # 3. Save Metrics to CSV
    metrics_rows = [
        {'head': 'cohort', 'metric': 'total_test_patients', 'value': len(sequences)},
        # Regression (N=150)
        {'head': 'regression_ctdna_30d', 'metric': 'mae', 'value': round(float(mae), 4)},
        {'head': 'regression_ctdna_30d', 'metric': 'rmse', 'value': round(float(rmse), 4)},
        {'head': 'regression_ctdna_30d', 'metric': 'r2_score', 'value': round(float(r2), 4)},
        {'head': 'regression_ctdna_30d', 'metric': 'pearson_r', 'value': round(float(p_r), 4)},
        {'head': 'regression_ctdna_30d', 'metric': 'pearson_pval', 'value': float(p_pval)},
        {'head': 'regression_ctdna_30d', 'metric': 'spearman_rho', 'value': round(float(s_rho), 4)},
        {'head': 'regression_ctdna_30d', 'metric': 'spearman_pval', 'value': float(s_pval)},
        {'head': 'regression_ctdna_30d', 'metric': 'mape_percent', 'value': round(float(mape), 2)},
        # Regression (N=141 subset)
        {'head': 'regression_ctdna_30d_subset_141', 'metric': 'mae_subset_141', 'value': round(float(mae_141), 4)},
        {'head': 'regression_ctdna_30d_subset_141', 'metric': 'rmse_subset_141', 'value': round(float(rmse_141), 4)},
        {'head': 'regression_ctdna_30d_subset_141', 'metric': 'r2_score_subset_141', 'value': round(float(r2_141), 4)},
        # Classification
        {'head': 'classification_progression', 'metric': 'accuracy', 'value': round(float(acc), 4)},
        {'head': 'classification_progression', 'metric': 'balanced_accuracy', 'value': round(float(bacc), 4)},
        {'head': 'classification_progression', 'metric': 'precision', 'value': round(float(p), 4)},
        {'head': 'classification_progression', 'metric': 'recall', 'value': round(float(r), 4)},
        {'head': 'classification_progression', 'metric': 'f1_score', 'value': round(float(f1), 4)},
        {'head': 'classification_progression', 'metric': 'roc_auc', 'value': round(float(roc_auc), 4)},
        {'head': 'classification_progression', 'metric': 'pr_auc', 'value': round(float(pr_auc), 4)},
        {'head': 'classification_progression', 'metric': 'true_negatives', 'value': int(cm[0, 0])},
        {'head': 'classification_progression', 'metric': 'false_positives', 'value': int(cm[0, 1])},
        {'head': 'classification_progression', 'metric': 'false_negatives', 'value': int(cm[1, 0])},
        {'head': 'classification_progression', 'metric': 'true_positives', 'value': int(cm[1, 1])},
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv_path = config.REPORTS_DIR / 'temporal_test_metrics.csv'
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Temporal test metrics saved to: {metrics_csv_path}")

    # 4. Plot Progression Confusion Matrix
    cm_fig_path = config.FIGURES_DIR / 'temporal_test_confusion_matrix.png'
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Non-Progressing (0)', 'Progressing (1)'], yticklabels=['Non-Progressing (0)', 'Progressing (1)'])
    plt.title('Temporal Test Set: Progression Trend Confusion Matrix', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=11)
    plt.ylabel('True Label', fontsize=11)
    plt.suptitle(f"{config.MANDATORY_DISCLAIMER}", fontsize=8, color='darkred', y=0.02)
    plt.tight_layout()
    plt.savefig(cm_fig_path, dpi=300)
    plt.close()
    print(f"Progression confusion matrix saved to: {cm_fig_path}")

    # 5. Plot Regression Scatter & Residual Plot
    scatter_fig_path = config.FIGURES_DIR / 'temporal_regression_scatter.png'
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter: Predicted vs True
    axes[0].scatter(all_targets_ctdna, all_preds_ctdna, alpha=0.7, color='teal', edgecolors='k', s=45)
    lims = [0, max(all_targets_ctdna.max(), all_preds_ctdna.max()) + 0.5]
    axes[0].plot(lims, lims, '--', color='red', label='Ideal Identity')
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].set_title(f'Predicted vs Actual ctDNA VAF 30d (R²={r2:.3f}, r={p_r:.3f})', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Ground Truth ctDNA VAF (%)', fontsize=10)
    axes[0].set_ylabel('Predicted ctDNA VAF (%)', fontsize=10)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Residuals: Residual vs True
    residuals = all_preds_ctdna - all_targets_ctdna
    axes[1].scatter(all_targets_ctdna, residuals, alpha=0.7, color='crimson', edgecolors='k', s=45)
    axes[1].axhline(0, linestyle='--', color='black', alpha=0.7)
    axes[1].set_title(f'Residual Analysis (MAE={mae:.3f}, RMSE={rmse:.3f})', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Ground Truth ctDNA VAF (%)', fontsize=10)
    axes[1].set_ylabel('Residual (Predicted - Actual)', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Temporal BiLSTM ctDNA 30-Day Regression on Locked Test Set (N={len(sequences)} patients)\n"
        f"{config.MANDATORY_DISCLAIMER}",
        fontsize=9, color='darkred', y=0.98
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(scatter_fig_path, dpi=300)
    plt.close()
    print(f"Regression scatter figure saved to: {scatter_fig_path}")

    return {
        'regression': {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'pearson_r': float(p_r),
            'spearman_rho': float(s_rho),
            'mape': float(mape),
            'mae_141': float(mae_141),
            'r2_141': float(r2_141)
        },
        'classification': {
            'accuracy': float(acc),
            'balanced_accuracy': float(bacc),
            'precision': float(p),
            'recall': float(r),
            'f1': float(f1),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'confusion_matrix': cm.tolist()
        },
        'total_patients': len(sequences)
    }

if __name__ == '__main__':
    evaluate_temporal_model()
