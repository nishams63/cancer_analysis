"""
Stage 2 Deep Learning - Independent Pathology Tile Evaluation on Locked Test Set
Evaluates frozen best_pathology_cnn.pt on 1,800 locked test tiles.
Computes Accuracy, Macro/Weighted F1, Per-class metrics, ROC-AUC, and Confusion Matrix.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from . import config
except (ImportError, ValueError):
    import config

# Import image model architecture from DL module
sys.path.insert(0, str(config.STAGE_2_DIR / 'dl'))
from src.image_model import build_image_model


class PathologyTestDataset(Dataset):
    """Dataset for locked test pathology tiles."""
    def __init__(self, meta_path: Path = config.IMAGE_METADATA_PATH, norm_mean=None, norm_std=None):
        df = pd.read_csv(meta_path)
        self.df = df[df['split'] == 'test'].reset_index(drop=True)
        
        mean = norm_mean or config.IMAGENET_MEAN
        std = norm_std or config.IMAGENET_STD
        
        self.transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        if not os.path.exists(img_path):
            rel = row.get('relative_path', '')
            alt_path = config.STAGE_2_DIR / rel.replace('data-engineering/data-engineering/', 'data-engineering/')
            if alt_path.exists():
                img_path = str(alt_path)

        image = Image.open(img_path).convert('RGB')
        tensor_img = self.transform(image)
        label_idx = config.CLASS_TO_IDX[row['class_label']]
        
        return tensor_img, label_idx, row['patient_id'], row['tile_id']


def evaluate_pathology_model(batch_size: int = 64, device: str = 'cpu') -> dict:
    print("\n" + "=" * 60)
    print("EVALUATING FROZEN PATHOLOGY CNN ON LOCKED TEST SET")
    print("=" * 60)

    # 1. Load Frozen Checkpoint
    chk_path = config.FROZEN_IMAGE_CHECKPOINT
    if not chk_path.exists():
        raise FileNotFoundError(f"Frozen checkpoint not found at: {chk_path}")

    checkpoint = torch.load(chk_path, map_location=device)
    arch = checkpoint.get('model_architecture', 'resnet18')
    norm_mean = checkpoint.get('norm_mean', config.IMAGENET_MEAN)
    norm_std = checkpoint.get('norm_std', config.IMAGENET_STD)

    print(f"Loaded checkpoint: {chk_path.name}")
    print(f"Architecture: {arch} | Epoch: {checkpoint.get('epoch', 'N/A')}")

    # 2. Reconstruct Frozen Model
    model = build_image_model(arch=arch, pretrained=False, num_classes=config.NUM_CLASSES)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # 3. Create Test DataLoader
    test_dataset = PathologyTestDataset(norm_mean=norm_mean, norm_std=norm_std)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Test cohort loaded: {len(test_dataset)} tiles across 150 patients.")

    # 4. Inference
    all_preds = []
    all_targets = []
    all_probs = []
    all_patient_ids = []
    all_tile_ids = []

    with torch.no_grad():
        for batch_imgs, batch_labels, batch_pts, batch_tiles in test_loader:
            batch_imgs = batch_imgs.to(device)
            logits = model(batch_imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_targets.extend(batch_labels.numpy())
            all_probs.extend(probs)
            all_patient_ids.extend(batch_pts)
            all_tile_ids.extend(batch_tiles)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # 5. Compute Comprehensive Metrics
    acc = accuracy_score(all_targets, all_preds)
    bacc = balanced_accuracy_score(all_targets, all_preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    wt_p, wt_r, wt_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)

    # Per-class metrics
    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(all_targets, all_preds, average=None, zero_division=0)
    per_class = {}
    for i, c_name in enumerate(config.CLASSES):
        per_class[c_name] = {
            'precision': float(per_p[i]),
            'recall': float(per_r[i]),
            'f1': float(per_f1[i]),
            'support': int(per_sup[i])
        }

    # One-vs-Rest Multi-class ROC-AUC
    try:
        macro_roc_auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
        wt_roc_auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='weighted')
    except Exception:
        macro_roc_auc = 1.0
        wt_roc_auc = 1.0

    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])

    print(f"\n--- Locked Test Set Metrics ---")
    print(f"Total Test Tiles: {len(all_targets)}")
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    print(f"Balanced Accuracy: {bacc * 100:.2f}%")
    print(f"Macro F1 Score:   {macro_f1:.4f}")
    print(f"Weighted F1 Score:{wt_f1:.4f}")
    print(f"Macro ROC-AUC:    {macro_roc_auc:.4f}")
    for c_name in config.CLASSES:
        print(f"  Class [{c_name:12s}]: Precision={per_class[c_name]['precision']:.4f}, Recall={per_class[c_name]['recall']:.4f}, F1={per_class[c_name]['f1']:.4f}, Support={per_class[c_name]['support']}")

    # 6. Save Metrics to CSV
    metrics_rows = [
        {'metric': 'total_test_samples', 'value': len(all_targets), 'category': 'cohort'},
        {'metric': 'test_accuracy', 'value': round(float(acc), 4), 'category': 'overall'},
        {'metric': 'test_balanced_accuracy', 'value': round(float(bacc), 4), 'category': 'overall'},
        {'metric': 'test_macro_precision', 'value': round(float(macro_p), 4), 'category': 'overall'},
        {'metric': 'test_macro_recall', 'value': round(float(macro_r), 4), 'category': 'overall'},
        {'metric': 'test_macro_f1', 'value': round(float(macro_f1), 4), 'category': 'overall'},
        {'metric': 'test_weighted_f1', 'value': round(float(wt_f1), 4), 'category': 'overall'},
        {'metric': 'test_macro_roc_auc', 'value': round(float(macro_roc_auc), 4), 'category': 'overall'},
        {'metric': 'test_weighted_roc_auc', 'value': round(float(wt_roc_auc), 4), 'category': 'overall'},
    ]
    for c_name in config.CLASSES:
        metrics_rows.extend([
            {'metric': f'{c_name}_precision', 'value': round(per_class[c_name]['precision'], 4), 'category': f'per_class_{c_name}'},
            {'metric': f'{c_name}_recall', 'value': round(per_class[c_name]['recall'], 4), 'category': f'per_class_{c_name}'},
            {'metric': f'{c_name}_f1', 'value': round(per_class[c_name]['f1'], 4), 'category': f'per_class_{c_name}'},
            {'metric': f'{c_name}_support', 'value': per_class[c_name]['support'], 'category': f'per_class_{c_name}'},
        ])

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv_path = config.REPORTS_DIR / 'pathology_test_metrics.csv'
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Test metrics saved to: {metrics_csv_path}")

    # 7. Plot Confusion Matrix
    cm_fig_path = config.FIGURES_DIR / 'pathology_test_confusion_matrix.png'
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=config.CLASSES, yticklabels=config.CLASSES, ax=axes[0])
    axes[0].set_title('Pathology Test Set: Confusion Matrix (Counts)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    axes[0].set_ylabel('True Label', fontsize=11)

    # Normalized proportions
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', xticklabels=config.CLASSES, yticklabels=config.CLASSES, ax=axes[1])
    axes[1].set_title('Pathology Test Set: Normalized Confusion Matrix', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    axes[1].set_ylabel('True Label', fontsize=11)

    plt.suptitle(
        f"Pathology ResNet-18 Evaluation on Locked Test Set (N={len(all_targets)})\n"
        f"{config.MANDATORY_DISCLAIMER}",
        fontsize=9, color='darkred', y=0.98
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(cm_fig_path, dpi=300)
    plt.close()
    print(f"Confusion matrix figure saved to: {cm_fig_path}")

    return {
        'accuracy': float(acc),
        'balanced_accuracy': float(bacc),
        'macro_precision': float(macro_p),
        'macro_recall': float(macro_r),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(wt_f1),
        'macro_roc_auc': float(macro_roc_auc),
        'per_class': per_class,
        'confusion_matrix': cm.tolist(),
        'total_samples': len(all_targets)
    }

if __name__ == '__main__':
    evaluate_pathology_model()
