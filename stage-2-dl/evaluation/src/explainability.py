"""
Stage 2 Deep Learning - Explainability & Interpretability Module
Vision: Grad-CAM on ResNet-18 layer4 across pathology classes.
Temporal: Permutation Feature Importance across 13 longitudinal features.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    from . import config, evaluate_temporal
except (ImportError, ValueError):
    import config, evaluate_temporal

sys.path.insert(0, str(config.STAGE_2_DIR / 'dl'))
from src.image_model import build_image_model
from src.temporal_model import build_temporal_model


# =====================================================================
# 1. VISION: GRAD-CAM ON RESNET-18
# =====================================================================

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        # Global average pool of gradients across spatial dims (H, W)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        # Resize to input tensor size (224, 224)
        cam_pil = Image.fromarray(np.uint8(224 * cam)).resize(config.IMAGE_SIZE, Image.BILINEAR)
        return np.array(cam_pil) / 255.0


def run_pathology_gradcam() -> Path:
    print("\n--- Running Pathology Grad-CAM Explainability ---")
    chk_path = config.FROZEN_IMAGE_CHECKPOINT
    checkpoint = torch.load(chk_path, map_location='cpu')
    model = build_image_model(arch='resnet18', pretrained=False, num_classes=config.NUM_CLASSES)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Target final residual layer
    target_layer = model.backbone.layer4[-1]
    grad_cam = GradCAM(model, target_layer)

    img_df = pd.read_csv(config.IMAGE_METADATA_PATH)
    test_df = img_df[img_df['split'] == 'test'].reset_index(drop=True)

    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])

    # Select 2 distinct tiles per class (6 total)
    selected_samples = []
    for c_name in config.CLASSES:
        c_sub = test_df[test_df['class_label'] == c_name].iloc[:2]
        for _, row in c_sub.iterrows():
            selected_samples.append(row)

    fig, axes = plt.subplots(len(selected_samples), 3, figsize=(11, 3.2 * len(selected_samples)))

    for i, row in enumerate(selected_samples):
        img_path = row['image_path']
        if not os.path.exists(img_path):
            rel = row.get('relative_path', '')
            alt_path = config.STAGE_2_DIR / rel.replace('data-engineering/data-engineering/', 'data-engineering/')
            img_path = str(alt_path)

        orig_pil = Image.open(img_path).convert('RGB')
        input_t = transform(orig_pil).unsqueeze(0)
        c_label = row['class_label']
        c_idx = config.CLASS_TO_IDX[c_label]

        cam_mask = grad_cam.generate(input_t, c_idx)

        # Colormap overlay
        heatmap = plt.get_cmap('jet')(cam_mask)[:, :, :3]
        orig_np = np.array(orig_pil.resize(config.IMAGE_SIZE)) / 255.0
        overlay = 0.55 * orig_np + 0.45 * heatmap
        overlay = np.clip(overlay, 0.0, 1.0)

        axes[i, 0].imshow(orig_np)
        axes[i, 0].set_title(f"Original Tile ({c_label})\nPatient: {row['patient_id']}", fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(cam_mask, cmap='jet')
        axes[i, 1].set_title(f"Grad-CAM Heatmap (layer4)", fontsize=10)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"Overlay on Morphology", fontsize=10)
        axes[i, 2].axis('off')

    plt.suptitle(
        f"ResNet-18 Grad-CAM Interpretability across Locked Test Pathology Tiles\n"
        f"{config.MANDATORY_DISCLAIMER}",
        fontsize=9, color='darkred', y=0.995
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    out_fig = config.FIGURES_DIR / 'pathology_gradcam_examples.png'
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Pathology Grad-CAM figure saved to: {out_fig}")
    return out_fig


# =====================================================================
# 2. TEMPORAL: PERMUTATION FEATURE IMPORTANCE
# =====================================================================

def run_temporal_feature_importance(n_repeats: int = 5) -> Path:
    print("\n--- Running Temporal Permutation Feature Importance ---")
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
    N = len(sequences)

    # 1. Baseline Predictions
    base_preds_ctdna = []
    base_preds_prog = []
    base_true_ctdna = []
    base_true_prog = []

    with torch.no_grad():
        for seq in sequences:
            x = seq['features'].unsqueeze(0)
            l = torch.tensor([seq['length']], dtype=torch.long)
            p_c, p_l = model(x, l)
            base_preds_ctdna.append(max(0.0, float(p_c.item())))
            base_preds_prog.append(1 if torch.sigmoid(p_l).item() >= 0.5 else 0)
            base_true_ctdna.append(seq['target_ctdna'])
            base_true_prog.append(seq['target_progression'])

    base_mae = np.mean(np.abs(np.array(base_true_ctdna) - np.array(base_preds_ctdna)))
    base_acc = np.mean(np.array(base_true_prog) == np.array(base_preds_prog))

    print(f"Baseline Test MAE: {base_mae:.4f} | Baseline Test Acc: {base_acc*100:.1f}%")

    # 2. Permutation Importance
    rng = np.random.RandomState(config.RANDOM_SEED)
    importance_mae = []
    importance_acc = []

    for f_idx, feat_name in enumerate(features_list):
        mae_drops = []
        acc_drops = []

        for _ in range(n_repeats):
            # Extract feature column across all patients at valid positions and shuffle
            perm_idx = rng.permutation(N)
            perm_preds_ctdna = []
            perm_preds_prog = []

            for i, seq in enumerate(sequences):
                # Replace feature column with permuted patient's feature column
                donor_seq = sequences[perm_idx[i]]
                x_perm = seq['features'].clone()
                donor_len = min(seq['length'], donor_seq['length'])
                x_perm[:donor_len, f_idx] = donor_seq['features'][:donor_len, f_idx]

                x_t = x_perm.unsqueeze(0)
                l_t = torch.tensor([seq['length']], dtype=torch.long)
                with torch.no_grad():
                    p_c, p_l = model(x_t, l_t)
                    perm_preds_ctdna.append(max(0.0, float(p_c.item())))
                    perm_preds_prog.append(1 if torch.sigmoid(p_l).item() >= 0.5 else 0)

            p_mae = np.mean(np.abs(np.array(base_true_ctdna) - np.array(perm_preds_ctdna)))
            p_acc = np.mean(np.array(base_true_prog) == np.array(perm_preds_prog))

            mae_drops.append(p_mae - base_mae)
            acc_drops.append(base_acc - p_acc)

        mean_mae_drop = float(np.mean(mae_drops))
        mean_acc_drop = float(np.mean(acc_drops))
        importance_mae.append(mean_mae_drop)
        importance_acc.append(mean_acc_drop)
        print(f"  Feature [{feat_name:20s}]: dMAE = +{mean_mae_drop:.4f}, dAcc = -{mean_acc_drop*100:.2f}%")

    # 3. Plot Permutation Feature Importance
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    y_pos = np.arange(len(features_list))
    sorted_mae_idx = np.argsort(importance_mae)
    sorted_acc_idx = np.argsort(importance_acc)

    # Plot A: Regression
    axes[0].barh(y_pos, [importance_mae[i] for i in sorted_mae_idx], color='teal', edgecolor='black')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([features_list[i] for i in sorted_mae_idx], fontsize=9)
    axes[0].set_xlabel('Increase in MAE (ΔMAE when permuted)', fontsize=10)
    axes[0].set_title('Head A: ctDNA 30-Day VAF Regression Feature Importance', fontsize=11, fontweight='bold')
    axes[0].grid(True, axis='x', alpha=0.3)

    # Plot B: Classification
    axes[1].barh(y_pos, [importance_acc[i] for i in sorted_acc_idx], color='coral', edgecolor='black')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([features_list[i] for i in sorted_acc_idx], fontsize=9)
    axes[1].set_xlabel('Drop in Accuracy (ΔAccuracy when permuted)', fontsize=10)
    axes[1].set_title('Head B: Progression Trend Classification Feature Importance', fontsize=11, fontweight='bold')
    axes[1].grid(True, axis='x', alpha=0.3)

    plt.suptitle(
        f"Temporal Permutation Feature Importance on Locked Test Cohort (N=150)\n"
        f"{config.MANDATORY_DISCLAIMER}",
        fontsize=9, color='darkred', y=0.98
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    out_fig = config.FIGURES_DIR / 'temporal_feature_importance.png'
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Temporal feature importance figure saved to: {out_fig}")
    return out_fig


def run_explainability():
    run_pathology_gradcam()
    run_temporal_feature_importance()

if __name__ == '__main__':
    run_explainability()
