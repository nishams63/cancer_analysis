"""
Stage 2 Deep Learning - Robustness & Perturbation Stress Testing Module
Tests frozen models under simulated clinical imaging artifacts and biomarker measurement noise.
Outputs:
  - reports/robustness_results.csv
  - figures/pathology_robustness.png
  - figures/temporal_robustness.png
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

try:
    from . import config, evaluate_temporal
except (ImportError, ValueError):
    import config, evaluate_temporal

sys.path.insert(0, str(config.STAGE_2_DIR / 'dl'))
from src.image_model import build_image_model
from src.temporal_model import build_temporal_model


# =====================================================================
# 1. VISION ROBUSTNESS BENCHMARK
# =====================================================================

def evaluate_perturbed_image_cohort(model, test_df, perturbation_type: str, severity: float, norm_mean, norm_std) -> dict:
    """Evaluates image model on test set under a specified perturbation."""
    preds, targets = [], []
    base_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    for _, row in test_df.iterrows():
        img_path = row['image_path']
        if not os.path.exists(img_path):
            rel = row.get('relative_path', '')
            alt = config.STAGE_2_DIR / rel.replace('data-engineering/data-engineering/', 'data-engineering/')
            img_path = str(alt)

        img = Image.open(img_path).convert('RGB')

        # Apply perturbation
        if perturbation_type == 'gaussian_blur':
            img = img.filter(ImageFilter.GaussianBlur(radius=severity))
        elif perturbation_type == 'brightness_jitter':
            img = TF.adjust_brightness(img, brightness_factor=1.0 + severity)
        elif perturbation_type == 'contrast_jitter':
            img = TF.adjust_contrast(img, contrast_factor=1.0 + severity)
        elif perturbation_type == 'resolution_downsample':
            # severity = scale factor (e.g. 0.5 = downsample to 50% then upsample)
            new_size = (int(config.IMAGE_SIZE[0] * severity), int(config.IMAGE_SIZE[1] * severity))
            img = img.resize(new_size, Image.BILINEAR).resize(config.IMAGE_SIZE, Image.BILINEAR)

        tensor_img = base_transform(img)

        # Additive pixel noise if selected
        if perturbation_type == 'gaussian_noise':
            noise = torch.randn_like(tensor_img) * severity
            tensor_img = tensor_img + noise

        with torch.no_grad():
            logit = model(tensor_img.unsqueeze(0))
            pred = int(torch.argmax(logit, dim=1).item())

        preds.append(pred)
        targets.append(config.CLASS_TO_IDX[row['class_label']])

    acc = accuracy_score(targets, preds)
    _, _, f1, _ = precision_recall_fscore_support(targets, preds, average='macro', zero_division=0)
    return {'accuracy': float(acc), 'macro_f1': float(f1)}


def run_pathology_robustness_audit(sample_size: int = 300) -> list:
    print("\n--- Running Pathology Vision Robustness Stress Test ---")
    chk_path = config.FROZEN_IMAGE_CHECKPOINT
    checkpoint = torch.load(chk_path, map_location='cpu')
    model = build_image_model(arch='resnet18', pretrained=False, num_classes=config.NUM_CLASSES)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    norm_mean = checkpoint.get('norm_mean', config.IMAGENET_MEAN)
    norm_std = checkpoint.get('norm_std', config.IMAGENET_STD)

    img_df = pd.read_csv(config.IMAGE_METADATA_PATH)
    test_df = img_df[img_df['split'] == 'test'].sample(n=min(sample_size, len(img_df[img_df['split'] == 'test'])), random_state=config.RANDOM_SEED)

    experiments = [
        ('baseline', 0.0, 'Baseline (No Perturbation)'),
        ('gaussian_blur', 1.0, 'Gaussian Blur (sigma=1.0)'),
        ('gaussian_blur', 2.5, 'Gaussian Blur (sigma=2.5)'),
        ('gaussian_noise', 0.1, 'Gaussian Pixel Noise (sigma=0.10)'),
        ('gaussian_noise', 0.25, 'Gaussian Pixel Noise (sigma=0.25)'),
        ('brightness_jitter', 0.25, 'Brightness Increase (+25%)'),
        ('brightness_jitter', -0.25, 'Brightness Decrease (-25%)'),
        ('contrast_jitter', 0.25, 'Contrast Increase (+25%)'),
        ('contrast_jitter', -0.25, 'Contrast Decrease (-25%)'),
        ('resolution_downsample', 0.5, 'Downsample 2x (112x112)'),
        ('resolution_downsample', 0.25, 'Downsample 4x (56x56)'),
    ]

    results = []
    for p_type, sev, label in experiments:
        res = evaluate_perturbed_image_cohort(model, test_df, p_type, sev, norm_mean, norm_std)
        results.append({
            'modality': 'image',
            'perturbation': label,
            'perturbation_type': p_type,
            'severity': sev,
            'accuracy': round(res['accuracy'], 4),
            'macro_f1': round(res['macro_f1'], 4),
            'metric_a_name': 'accuracy',
            'metric_a_value': round(res['accuracy'], 4),
            'metric_b_name': 'macro_f1',
            'metric_b_value': round(res['macro_f1'], 4)
        })
        print(f"  {label:35s}: Acc = {res['accuracy']*100:.1f}%, Macro F1 = {res['macro_f1']:.4f}")

    # Plot Pathology Robustness
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [r['perturbation'] for r in results]
    f1_scores = [r['macro_f1'] for r in results]
    acc_scores = [r['accuracy'] for r in results]

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, acc_scores, width, label='Accuracy', color='royalblue')
    ax.bar(x + width/2, f1_scores, width, label='Macro F1', color='darkorange')

    ax.set_ylabel('Score (0.0 - 1.0)', fontsize=11)
    ax.set_title('Pathology CNN (ResNet-18) Robustness Under Physical & Stain Perturbations', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylim([0.0, 1.05])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle(f"{config.MANDATORY_DISCLAIMER}", fontsize=8, color='darkred', y=0.01)
    plt.tight_layout()
    out_fig = config.FIGURES_DIR / 'pathology_robustness.png'
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Pathology robustness figure saved to: {out_fig}")
    return results


# =====================================================================
# 2. TEMPORAL ROBUSTNESS BENCHMARK
# =====================================================================

def evaluate_perturbed_temporal_cohort(model, sequences, perturbation_type: str, severity: float) -> dict:
    """Evaluates temporal model under simulated clinical measurement noise and missingness."""
    rng = np.random.RandomState(config.RANDOM_SEED)
    preds_ctdna, true_ctdna = [], []
    preds_prog, true_prog = [], []

    with torch.no_grad():
        for seq in sequences:
            x = seq['features'].clone()
            seq_len = seq['length']

            # Apply perturbation
            if perturbation_type == 'biomarker_noise':
                # Add Gaussian noise to continuous normalized features (columns 0:8)
                noise = torch.randn_like(x[:seq_len, :8]) * severity
                x[:seq_len, :8] = x[:seq_len, :8] + noise
            elif perturbation_type == 'missingness_spike':
                # Randomly drop timepoints by masking out features
                for t in range(seq_len):
                    if rng.rand() < severity:
                        x[t, :8] = 0.0  # Zero imputation
                        x[t, 8:] = 1.0  # Mark missing
            elif perturbation_type == 'trajectory_truncation':
                # Retain only the first `int(severity)` visits
                trunc_len = max(1, min(int(severity), seq_len))
                x = x[:trunc_len, :]
                seq_len = trunc_len

            x_t = x.unsqueeze(0)
            lens = torch.tensor([seq_len], dtype=torch.long)
            p_c, p_l = model(x_t, lens)

            preds_ctdna.append(max(0.0, float(p_c.item())))
            true_ctdna.append(seq['target_ctdna'])
            preds_prog.append(1 if torch.sigmoid(p_l).item() >= 0.5 else 0)
            true_prog.append(seq['target_progression'])

    mae = mean_absolute_error(true_ctdna, preds_ctdna)
    rmse = np.sqrt(mean_squared_error(true_ctdna, preds_ctdna))
    acc = accuracy_score(true_prog, preds_prog)
    _, _, f1, _ = precision_recall_fscore_support(true_prog, preds_prog, average='binary', zero_division=0)
    return {'mae': float(mae), 'rmse': float(rmse), 'accuracy': float(acc), 'f1': float(f1)}


def run_temporal_robustness_audit() -> list:
    print("\n--- Running Temporal Longitudinal Robustness Stress Test ---")
    chk_path = config.FROZEN_TEMPORAL_CHECKPOINT
    checkpoint = torch.load(chk_path, map_location='cpu')
    arch = checkpoint.get('model_architecture', 'lstm')
    hidden_dim = checkpoint.get('hidden_dim', 64)
    norm_params = checkpoint.get('norm_params', {})
    features_list = checkpoint.get('temporal_features', config.get_checkpoint_temporal_features())

    model = build_temporal_model(arch=arch, input_dim=len(features_list), hidden_dim=hidden_dim)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    sequences = evaluate_temporal.prepare_test_patient_sequences(config.BIOMARKERS_PATH, norm_params, features_list)

    experiments = [
        ('baseline', 0.0, 'Baseline (No Noise)'),
        ('biomarker_noise', 0.10, 'Biomarker Assay Noise (10%)'),
        ('biomarker_noise', 0.25, 'Biomarker Assay Noise (25%)'),
        ('biomarker_noise', 0.50, 'Biomarker Assay Noise (50%)'),
        ('missingness_spike', 0.15, 'Missingness Spike (15% drop)'),
        ('missingness_spike', 0.30, 'Missingness Spike (30% drop)'),
        ('missingness_spike', 0.50, 'Missingness Spike (50% drop)'),
        ('trajectory_truncation', 3.0, 'Truncated Trajectory (First 3 visits)'),
        ('trajectory_truncation', 2.0, 'Truncated Trajectory (First 2 visits)'),
    ]

    results = []
    for p_type, sev, label in experiments:
        res = evaluate_perturbed_temporal_cohort(model, sequences, p_type, sev)
        results.append({
            'modality': 'temporal',
            'perturbation': label,
            'perturbation_type': p_type,
            'severity': sev,
            'mae': round(res['mae'], 4),
            'rmse': round(res['rmse'], 4),
            'accuracy': round(res['accuracy'], 4),
            'f1': round(res['f1'], 4),
            'metric_a_name': 'mae',
            'metric_a_value': round(res['mae'], 4),
            'metric_b_name': 'f1_progression',
            'metric_b_value': round(res['f1'], 4)
        })
        print(f"  {label:40s}: MAE = {res['mae']:.4f}, Prog F1 = {res['f1']:.4f}, Acc = {res['accuracy']*100:.1f}%")

    # Plot Temporal Robustness
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    labels = [r['perturbation'] for r in results]
    maes = [r['mae'] for r in results]
    f1s = [r['f1'] for r in results]

    y_pos = np.arange(len(labels))
    axes[0].barh(y_pos, maes, color='teal', edgecolor='black')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].set_xlabel('ctDNA 30d MAE (Lower is better)', fontsize=10)
    axes[0].set_title('Head A: ctDNA Regression Error Under Stress', fontsize=11, fontweight='bold')
    axes[0].grid(True, axis='x', alpha=0.3)

    axes[1].barh(y_pos, f1s, color='coral', edgecolor='black')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(labels, fontsize=9)
    axes[1].set_xlabel('Progression F1 Score (Higher is better)', fontsize=10)
    axes[1].set_title('Head B: Progression Classification F1 Under Stress', fontsize=11, fontweight='bold')
    axes[1].set_xlim([0.0, 1.05])
    axes[1].grid(True, axis='x', alpha=0.3)

    plt.suptitle(
        f"Longitudinal BiLSTM Robustness Under Simulated Clinical Degradation (N=150)\n"
        f"{config.MANDATORY_DISCLAIMER}",
        fontsize=9, color='darkred', y=0.98
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    out_fig = config.FIGURES_DIR / 'temporal_robustness.png'
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Temporal robustness figure saved to: {out_fig}")
    return results


def run_robustness_audit():
    img_res = run_pathology_robustness_audit()
    temp_res = run_temporal_robustness_audit()

    all_res = img_res + temp_res
    csv_path = config.REPORTS_DIR / 'robustness_results.csv'
    pd.DataFrame(all_res).to_csv(csv_path, index=False)
    print(f"Robustness results saved to: {csv_path}")

if __name__ == '__main__':
    run_robustness_audit()
