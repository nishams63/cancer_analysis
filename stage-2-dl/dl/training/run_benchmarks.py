"""
Stage 2 Deep Learning - Master Benchmark & Experiment Runner
SCIENTIFIC INTEGRITY VERSION — Zero fabricated results.

Executes:
  1. Controlled Pathology Benchmarks (ResNet-18, ResNet-50, EfficientNet-B0, ViT)
  2. Patient-Level Attention-MIL (Mean, Median, Max, Attention pooling)
  3. Longitudinal Temporal Models (Real BiLSTM baseline vs Continuous Temporal Transformer)
  4. Learned Multimodal Fusion (Fixed Linear, Concat MLP, Gated Fusion, Cross-Attention)
  5. Real Modality Ablation Experiments (8 distinct configurations, ALL actually evaluated)
  6. Clinical Challenge Robustness Suite (Visual shifts, Lab perturbations, Modality dropouts)
  7. Uncertainty (MC Dropout) & Calibration (Temperature Scaling, ECE, Brier)
  8. Latent OOD Shift Detection
  9. Real Multi-Seed Training (seeds 42, 123, 2026 — each actually trained)
 10. Generates all required CSVs and comprehensive Markdown reports

EVERY metric in this file comes from actual model training and evaluation.
NO arithmetic offsets. NO fabricated baselines. NO simulated seeds.
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, mean_absolute_error,
    root_mean_squared_error, r2_score, confusion_matrix
)

# Add stage-2-dl root to path
DL_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(DL_ROOT))

from dl import config, dataset_v2
from dl.models import pathology_benchmarks as pb
from dl.models import attention_mil as am
from dl.models import temporal_transformer as tt
from dl.models import multimodal_fusion as mf
from dl import uncertainty_calibration as uc
from dl import ood_detector as od
from dl import explainability as exp
from dl import robustness_suite as rs

# Also import baseline temporal model for real BiLSTM benchmark
from dl.src import temporal_model as baseline_temporal


def set_seed(seed: int):
    """Sets global random seeds for deterministic execution."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_cls_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None, is_multiclass: bool = False) -> Dict[str, float]:
    """Computes comprehensive classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    avg = 'macro' if is_multiclass else 'binary'
    prec = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
    rec = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

    roc_auc, pr_auc = float('nan'), float('nan')
    if y_prob is not None:
        if not np.isfinite(y_prob).all():
            raise ValueError("Non-finite predictions must not produce benchmark metrics")
        if is_multiclass:
            from sklearn.preprocessing import label_binarize
            labels = np.arange(y_prob.shape[1])
            if len(np.unique(y_true)) == len(labels):
                roc_auc = float(roc_auc_score(y_true, y_prob, multi_class='ovr', labels=labels))
                binary = label_binarize(y_true, classes=labels)
                areas = []
                for k in labels:
                    precision, recall, _ = precision_recall_curve(binary[:, k], y_prob[:, k])
                    areas.append(auc(recall, precision))
                pr_auc = float(np.mean(areas))
        elif len(np.unique(y_true)) == 2:
            roc_auc = float(roc_auc_score(y_true, y_prob))
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = float(auc(recall, precision))

    return {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
        'macro_f1': round(macro_f1, 4),
        'roc_auc': round(roc_auc, 4),
        'pr_auc': round(pr_auc, 4)
    }


def compute_reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes regression metrics for ctDNA forecasting."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'r2': round(r2, 4)
    }


def count_parameters(model: nn.Module) -> int:
    """Counts total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model: nn.Module) -> int:
    """Counts ALL parameters (trainable + frozen)."""
    return sum(p.numel() for p in model.parameters())


# =========================================================================
# 1. Pathology Benchmark Training & Evaluation
# =========================================================================

def train_pathology_model(
    model_name: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 5e-4,
    device: torch.device = torch.device('cpu'),
    sample_limit: Optional[int] = None,
    use_pretrained: bool = True
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any], float, float]:
    """
    Trains a pathology benchmark model with PRETRAINED weights and trainable head.
    Returns: (model, val_metrics, test_metrics, best_val_f1, training_time_seconds)
    """
    # FIX: Use pretrained=True so backbone features are meaningful
    model = pb.build_pathology_benchmark_model(
        model_name, pretrained=use_pretrained, freeze_backbone=use_pretrained
    ).to(device)

    chk_path = config.CHECKPOINTS_DIR / f'best_{model_name}.pt'
    if False:  # Legacy caches have no reproducible experiment fingerprint.
        checkpoint = torch.load(str(chk_path), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        train_time = checkpoint.get('training_time', 0.0)
        print(f"    [Cache Hit] Loaded {chk_path.name} (Val F1: {best_val_f1:.4f})")
        val_ds = dataset_v2.PathologyTileDataset(split='validation')
        test_ds = dataset_v2.PathologyTileDataset(split='test')
        if sample_limit:
            val_ds = Subset(val_ds, range(min(sample_limit // 2, len(val_ds))))
            test_ds = Subset(test_ds, range(min(sample_limit // 2, len(test_ds))))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model.eval()
        v_preds, v_targets, v_probs = [], [], []
        with torch.no_grad():
            for imgs, targets, _ in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                v_preds.extend(np.argmax(probs, axis=1))
                v_targets.extend(targets.numpy())
                v_probs.extend(probs)
        v_metrics = compute_cls_metrics(np.array(v_targets), np.array(v_preds), np.array(v_probs), is_multiclass=True)

        t_preds, t_targets, t_probs = [], [], []
        with torch.no_grad():
            for imgs, targets, _ in test_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                t_preds.extend(np.argmax(probs, axis=1))
                t_targets.extend(targets.numpy())
                t_probs.extend(probs)
        test_metrics = compute_cls_metrics(np.array(t_targets), np.array(t_preds), np.array(t_probs), is_multiclass=True)
        return model, v_metrics, test_metrics, best_val_f1, train_time

    train_ds = dataset_v2.PathologyTileDataset(split='train')
    val_ds = dataset_v2.PathologyTileDataset(split='validation')
    test_ds = dataset_v2.PathologyTileDataset(split='test')

    if sample_limit:
        train_ds = Subset(train_ds, range(min(sample_limit, len(train_ds))))
        val_ds = Subset(val_ds, range(min(sample_limit // 2, len(val_ds))))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1 = -1.0
    best_weights = None
    patience = 4
    patience_counter = 0

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for imgs, targets, _ in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation evaluation (for model selection)
        model.eval()
        v_preds, v_targets, v_probs = [], [], []
        with torch.no_grad():
            for imgs, targets, _ in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                v_preds.extend(preds)
                v_targets.extend(targets.numpy())
                v_probs.extend(probs)

        v_metrics = compute_cls_metrics(np.array(v_targets), np.array(v_preds), np.array(v_probs), is_multiclass=True)
        if v_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = v_metrics['macro_f1']
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch}")
                break

    train_time = time.time() - t_start

    # Load best validation checkpoint
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Save checkpoint
    torch.save({
        'model_name': model_name,
        'state_dict': model.state_dict(),
        'best_val_f1': best_val_f1,
        'num_params': count_parameters(model),
        'total_params': count_all_parameters(model),
        'training_time': train_time,
        'pretrained': use_pretrained,
        'disclaimer': config.MANDATORY_DISCLAIMER
    }, str(chk_path))

    # Single final evaluation on locked test set
    model.eval()
    t_preds, t_targets, t_probs = [], [], []
    with torch.no_grad():
        for imgs, targets, _ in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            t_preds.extend(preds)
            t_targets.extend(targets.numpy())
            t_probs.extend(probs)

    test_metrics = compute_cls_metrics(np.array(t_targets), np.array(t_preds), np.array(t_probs), is_multiclass=True)
    return model, v_metrics, test_metrics, best_val_f1, train_time


# =========================================================================
# 2. Patient-Level Attention-MIL Training & Comparison
# =========================================================================

def train_attention_mil(
    pooling_mode: str = 'attention',
    epochs: int = 8,
    device: torch.device = torch.device('cpu'),
    sample_limit: Optional[int] = None
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """Trains or evaluates PatientPathologyMIL with specified pooling strategy."""
    model = am.PatientPathologyMIL(backbone_name='resnet50', pooling_mode=pooling_mode, feature_dim=128, pretrained=True).to(device)

    chk_path = config.CHECKPOINTS_DIR / 'best_attention_mil.pt'
    if False:  # Do not accept unverified legacy checkpoints.
        checkpoint = torch.load(str(chk_path), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"    [Cache Hit] Loaded {chk_path.name}")
        val_bags = dataset_v2.PatientPathologyBagDataset(split='validation')
        test_bags = dataset_v2.PatientPathologyBagDataset(split='test')
        if sample_limit:
            val_bags = Subset(val_bags, range(min(sample_limit // 2, len(val_bags))))
            test_bags = Subset(test_bags, range(min(sample_limit, len(test_bags))))
        val_loader = DataLoader(val_bags, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_bags, batch_size=8, shuffle=False)

        model.eval()
        v_preds, v_targets, v_probs = [], [], []
        with torch.no_grad():
            for bag_imgs, pat_labels, _, _ in val_loader:
                bag_imgs = bag_imgs.to(device)
                logits, _, _ = model(bag_images=bag_imgs)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                v_preds.extend(np.argmax(probs, axis=1))
                v_targets.extend(pat_labels.numpy())
                v_probs.extend(probs)
        v_metrics = compute_cls_metrics(np.array(v_targets), np.array(v_preds), np.array(v_probs), is_multiclass=True)

        t_preds, t_targets, t_probs = [], [], []
        with torch.no_grad():
            for bag_imgs, pat_labels, _, _ in test_loader:
                bag_imgs = bag_imgs.to(device)
                logits, _, _ = model(bag_images=bag_imgs)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                t_preds.extend(np.argmax(probs, axis=1))
                t_targets.extend(pat_labels.numpy())
                t_probs.extend(probs)
        t_metrics = compute_cls_metrics(np.array(t_targets), np.array(t_preds), np.array(t_probs), is_multiclass=True)
        return model, v_metrics, t_metrics

    train_bags = dataset_v2.PatientPathologyBagDataset(split='train')
    val_bags = dataset_v2.PatientPathologyBagDataset(split='validation')
    test_bags = dataset_v2.PatientPathologyBagDataset(split='test')

    if sample_limit:
        train_bags = Subset(train_bags, range(min(sample_limit, len(train_bags))))
        val_bags = Subset(val_bags, range(min(sample_limit // 2, len(val_bags))))
        test_bags = Subset(test_bags, range(min(sample_limit, len(test_bags))))

    train_loader = DataLoader(train_bags, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_bags, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_bags, batch_size=8, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1 = -1.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        for bag_imgs, pat_labels, _, _ in train_loader:
            bag_imgs, pat_labels = bag_imgs.to(device), pat_labels.to(device)
            optimizer.zero_grad()
            logits, _, _ = model(bag_images=bag_imgs)
            loss = criterion(logits, pat_labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        v_preds, v_targets, v_probs = [], [], []
        with torch.no_grad():
            for bag_imgs, pat_labels, _, _ in val_loader:
                bag_imgs = bag_imgs.to(device)
                logits, _, _ = model(bag_images=bag_imgs)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                v_preds.extend(np.argmax(probs, axis=1))
                v_targets.extend(pat_labels.numpy())
                v_probs.extend(probs)

        v_metrics = compute_cls_metrics(np.array(v_targets), np.array(v_preds), np.array(v_probs), is_multiclass=True)
        if v_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = v_metrics['macro_f1']
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict(best_weights)

    if pooling_mode == 'attention':
        torch.save({'state_dict': model.state_dict(), 'best_val_f1': best_val_f1}, str(chk_path))

    # Test evaluation
    model.eval()
    t_preds, t_targets, t_probs = [], [], []
    with torch.no_grad():
        for bag_imgs, pat_labels, _, _ in test_loader:
            bag_imgs = bag_imgs.to(device)
            logits, _, _ = model(bag_images=bag_imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            t_preds.extend(np.argmax(probs, axis=1))
            t_targets.extend(pat_labels.numpy())
            t_probs.extend(probs)

    t_metrics = compute_cls_metrics(np.array(t_targets), np.array(t_preds), np.array(t_probs), is_multiclass=True)
    return model, v_metrics, t_metrics


# =========================================================================
# 3a. REAL BiLSTM Baseline Training & Evaluation
# =========================================================================

def train_bilstm(
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    lambda_cls: float = 1.0,
    device: torch.device = torch.device('cpu')
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any], Dict[str, Tuple[float, float]], float]:
    """
    Trains the REAL baseline BiLSTM model from src/temporal_model.py.
    Uses identical patient splits, normalization, and evaluation protocol as the Transformer.
    Returns: (model, val_metrics, test_metrics, norm_params, training_time)
    """
    train_ds = dataset_v2.ContinuousTemporalSequenceDataset(split='train')
    val_ds = dataset_v2.ContinuousTemporalSequenceDataset(split='validation', norm_params=train_ds.norm_params)
    test_ds = dataset_v2.ContinuousTemporalSequenceDataset(split='test', norm_params=train_ds.norm_params)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = baseline_temporal.BiLSTMForecaster(
        input_dim=config.NUM_TEMPORAL_FEATURES, hidden_dim=64, num_layers=2
    ).to(device)

    chk_path = config.CHECKPOINTS_DIR / 'best_bilstm_benchmark.pt'
    if False:  # Legacy caches have no reproducible experiment fingerprint.
        checkpoint = torch.load(str(chk_path), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        train_time = checkpoint.get('training_time', 0.0)
        print(f"    [Cache Hit] Loaded {chk_path.name}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        best_score = -999.0
        best_weights = None
        t_start = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            for batch in train_loader:
                feats = batch['features'].to(device)
                lens = batch['length'].to(device)
                y_ctdna = batch['target_ctdna'].to(device)
                y_prog = batch['target_progression'].to(device)

                optimizer.zero_grad()
                pred_ctdna, pred_logits = model(feats, lens)
                reg_loss = F.smooth_l1_loss(pred_ctdna, y_ctdna)
                cls_loss = F.binary_cross_entropy_with_logits(pred_logits, y_prog)
                total_loss = reg_loss + (lambda_cls * cls_loss)
                total_loss.backward()
                optimizer.step()
            scheduler.step()

            # Validation
            model.eval()
            all_y_c, all_p_c, all_y_p, all_p_p = [], [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    feats = batch['features'].to(device)
                    lens = batch['length'].to(device)
                    pred_c, pred_l = model(feats, lens)
                    all_y_c.extend(batch['target_ctdna'].numpy())
                    all_p_c.extend(pred_c.cpu().numpy())
                    all_y_p.extend(batch['target_progression'].numpy())
                    all_p_p.extend(torch.sigmoid(pred_l).cpu().numpy())

            v_cls = compute_cls_metrics(np.array(all_y_p), (np.array(all_p_p) >= 0.5).astype(int), np.array(all_p_p))
            v_reg = compute_reg_metrics(np.array(all_y_c), np.array(all_p_c))
            score = v_cls['f1'] - (v_reg['mae'] * 0.1)

            if score > best_score:
                best_score = score
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        train_time = time.time() - t_start

        if best_weights is not None:
            model.load_state_dict(best_weights)

        torch.save({
            'state_dict': model.state_dict(),
            'best_score': best_score,
            'norm_params': train_ds.norm_params,
            'num_params': count_parameters(model),
            'training_time': train_time,
            'disclaimer': config.MANDATORY_DISCLAIMER
        }, str(chk_path))

    # Evaluate on validation
    model.eval()
    all_y_c, all_p_c, all_y_p, all_p_p = [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            feats = batch['features'].to(device)
            lens = batch['length'].to(device)
            pred_c, pred_l = model(feats, lens)
            all_y_c.extend(batch['target_ctdna'].numpy())
            all_p_c.extend(pred_c.cpu().numpy())
            all_y_p.extend(batch['target_progression'].numpy())
            all_p_p.extend(torch.sigmoid(pred_l).cpu().numpy())
    v_reg = compute_reg_metrics(np.array(all_y_c), np.array(all_p_c))
    v_cls = compute_cls_metrics(np.array(all_y_p), (np.array(all_p_p) >= 0.5).astype(int), np.array(all_p_p))
    v_metrics = {**v_cls, **v_reg}

    # Evaluate on test
    all_y_c, all_p_c, all_y_p, all_p_p = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            feats = batch['features'].to(device)
            lens = batch['length'].to(device)
            pred_c, pred_l = model(feats, lens)
            all_y_c.extend(batch['target_ctdna'].numpy())
            all_p_c.extend(pred_c.cpu().numpy())
            all_y_p.extend(batch['target_progression'].numpy())
            all_p_p.extend(torch.sigmoid(pred_l).cpu().numpy())
    t_reg = compute_reg_metrics(np.array(all_y_c), np.array(all_p_c))
    t_cls = compute_cls_metrics(np.array(all_y_p), (np.array(all_p_p) >= 0.5).astype(int), np.array(all_p_p))
    t_metrics = {**t_cls, **t_reg}

    return model, v_metrics, t_metrics, train_ds.norm_params, train_time


# =========================================================================
# 3b. Temporal Transformer Training & Evaluation
# =========================================================================

def train_temporal_transformer(
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    lambda_cls: float = 1.0,
    device: torch.device = torch.device('cpu')
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any], Dict[str, Tuple[float, float]], float]:
    """Trains ContinuousTemporalTransformer with multi-task objective."""
    train_ds = dataset_v2.ContinuousTemporalSequenceDataset(split='train')
    val_ds = dataset_v2.ContinuousTemporalSequenceDataset(split='validation', norm_params=train_ds.norm_params)
    test_ds = dataset_v2.ContinuousTemporalSequenceDataset(split='test', norm_params=train_ds.norm_params)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = tt.ContinuousTemporalTransformer(d_model=64, nhead=4, num_layers=2).to(device)

    chk_path = config.CHECKPOINTS_DIR / 'best_temporal_transformer.pt'
    if False:  # Legacy caches have no reproducible experiment fingerprint.
        checkpoint = torch.load(str(chk_path), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        norm_params = checkpoint.get('norm_params', train_ds.norm_params)
        train_time = checkpoint.get('training_time', 0.0)
        print(f"    [Cache Hit] Loaded {chk_path.name}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        best_score = -999.0
        best_weights = None
        t_start = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            for batch in train_loader:
                feats = batch['features'].to(device)
                lens = batch['length'].to(device)
                y_ctdna = batch['target_ctdna'].to(device)
                y_prog = batch['target_progression'].to(device)

                optimizer.zero_grad()
                pred_ctdna, pred_logits, _ = model(feats, lens)
                loss, _, _ = model.compute_multi_task_loss(pred_ctdna, y_ctdna, pred_logits, y_prog, lambda_cls=lambda_cls)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Validation
            model.eval()
            all_y_c, all_p_c, all_y_p, all_p_p = [], [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    feats = batch['features'].to(device)
                    lens = batch['length'].to(device)
                    pred_c, pred_l, _ = model(feats, lens)
                    all_y_c.extend(batch['target_ctdna'].numpy())
                    all_p_c.extend(pred_c.cpu().numpy())
                    all_y_p.extend(batch['target_progression'].numpy())
                    all_p_p.extend(torch.sigmoid(pred_l).cpu().numpy())

            v_reg = compute_reg_metrics(np.array(all_y_c), np.array(all_p_c))
            v_cls = compute_cls_metrics(np.array(all_y_p), (np.array(all_p_p) >= 0.5).astype(int), np.array(all_p_p))
            score = v_cls['f1'] - (v_reg['mae'] * 0.1)

            if score > best_score:
                best_score = score
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        train_time = time.time() - t_start

        if best_weights is not None:
            model.load_state_dict(best_weights)

        torch.save({
            'state_dict': model.state_dict(),
            'best_score': best_score,
            'norm_params': train_ds.norm_params,
            'num_params': count_parameters(model),
            'training_time': train_time,
            'disclaimer': config.MANDATORY_DISCLAIMER
        }, str(chk_path))
        norm_params = train_ds.norm_params

    # Evaluate on validation
    model.eval()
    all_y_c, all_p_c, all_y_p, all_p_p = [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            feats = batch['features'].to(device)
            lens = batch['length'].to(device)
            pred_c, pred_l, _ = model(feats, lens)
            all_y_c.extend(batch['target_ctdna'].numpy())
            all_p_c.extend(pred_c.cpu().numpy())
            all_y_p.extend(batch['target_progression'].numpy())
            all_p_p.extend(torch.sigmoid(pred_l).cpu().numpy())
    v_reg = compute_reg_metrics(np.array(all_y_c), np.array(all_p_c))
    v_cls = compute_cls_metrics(np.array(all_y_p), (np.array(all_p_p) >= 0.5).astype(int), np.array(all_p_p))
    v_metrics = {**v_cls, **v_reg}

    # Evaluate on test
    all_y_c, all_p_c, all_y_p, all_p_p = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            feats = batch['features'].to(device)
            lens = batch['length'].to(device)
            pred_c, pred_l, _ = model(feats, lens)
            all_y_c.extend(batch['target_ctdna'].numpy())
            all_p_c.extend(pred_c.cpu().numpy())
            all_y_p.extend(batch['target_progression'].numpy())
            all_p_p.extend(torch.sigmoid(pred_l).cpu().numpy())
    t_reg = compute_reg_metrics(np.array(all_y_c), np.array(all_p_c))
    t_cls = compute_cls_metrics(np.array(all_y_p), (np.array(all_p_p) >= 0.5).astype(int), np.array(all_p_p))
    t_metrics = {**t_cls, **t_reg}

    return model, v_metrics, t_metrics, norm_params, train_time


# =========================================================================
# 4. Multimodal Dataset Preparation & Fusion Training
# =========================================================================

def extract_patient_embeddings(
    pathology_model: nn.Module,
    temporal_model: nn.Module,
    split: str,
    norm_params: Dict[str, Tuple[float, float]],
    device: torch.device = torch.device('cpu'),
    is_bilstm: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, List[str]]:
    """Extracts aligned patient pathology and temporal embeddings for multimodal modeling."""
    cache_file = config.RESULTS_DIR / f'embeddings_{split}.pt'
    if False:  # Do not accept unverified legacy embeddings.
        cached = torch.load(str(cache_file), map_location='cpu', weights_only=False)
        print(f"  [Cache Hit] Loaded aligned {split} embeddings from cache.")
        return cached['pathology_reprs'], cached['temporal_reprs'], cached['prog_targets'], cached['ctdna_targets'], cached['patient_ids']

    pathology_model.eval()
    temporal_model.eval()

    bag_ds = dataset_v2.PatientPathologyBagDataset(split=split)
    temp_ds = dataset_v2.ContinuousTemporalSequenceDataset(split=split, norm_params=norm_params)

    temp_map = {item['patient_id']: item for item in temp_ds}

    path_reprs, temp_reprs = [], []
    prog_targets, ctdna_targets = [], []
    patient_ids = []

    bag_loader = DataLoader(bag_ds, batch_size=8, shuffle=False)
    with torch.no_grad():
        for bag_imgs, _, _, p_ids in bag_loader:
            bag_imgs = bag_imgs.to(device)
            _, p_reps, _ = pathology_model(bag_images=bag_imgs)
            for b, pat_id in enumerate(p_ids):
                if pat_id not in temp_map:
                    continue
                t_item = temp_map[pat_id]
                t_feat = t_item['features'].unsqueeze(0).to(device)
                t_len = torch.tensor([t_item['length']], dtype=torch.long).to(device)

                if is_bilstm:
                    # BiLSTM returns (ctdna_pred, prog_logits) — no patient_repr
                    # Extract the hidden representation manually
                    out, _ = temporal_model.lstm(t_feat)
                    last_idx = (t_len - 1).clamp(min=0).unsqueeze(1).unsqueeze(2).expand(-1, 1, out.size(2))
                    t_rep = out.gather(1, last_idx).squeeze(1)
                else:
                    _, _, t_rep = temporal_model(t_feat, t_len)

                path_reprs.append(p_reps[b].cpu())
                temp_reprs.append(t_rep.squeeze(0).cpu())
                prog_targets.append(t_item['target_progression'].item())
                ctdna_targets.append(t_item['target_ctdna'].item())
                patient_ids.append(pat_id)

    p_tensor = torch.stack(path_reprs)
    t_tensor = torch.stack(temp_reprs)
    p_targets = np.array(prog_targets)
    c_targets = np.array(ctdna_targets)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        'pathology_reprs': p_tensor,
        'temporal_reprs': t_tensor,
        'prog_targets': p_targets,
        'ctdna_targets': c_targets,
        'patient_ids': patient_ids
    }, str(cache_file))

    return p_tensor, t_tensor, p_targets, c_targets, patient_ids


def train_multimodal_fusion(
    fusion_type: str,
    train_data: Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray],
    val_data: Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray],
    test_data: Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray],
    epochs: int = 15,
    lr: float = 1e-3,
    device: torch.device = torch.device('cpu'),
    seed: int = 42
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """Trains and validates a learned multimodal fusion model."""
    set_seed(seed)
    p_tr, t_tr, y_prog_tr, y_ctdna_tr = train_data
    p_va, t_va, y_prog_va, y_ctdna_va = val_data
    p_te, t_te, y_prog_te, y_ctdna_te = test_data

    model = mf.build_fusion_model(fusion_type, pathology_dim=p_tr.shape[1], temporal_dim=t_tr.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    y_p_tr_t = torch.tensor(y_prog_tr, dtype=torch.float32)
    y_c_tr_t = torch.tensor(y_ctdna_tr, dtype=torch.float32)

    best_val_f1 = -1.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(p_tr.to(device), t_tr.to(device))
        loss_reg = F.smooth_l1_loss(out['ctdna_forecast'], y_c_tr_t.to(device))
        loss_cls = F.binary_cross_entropy_with_logits(out['progression_logits'], y_p_tr_t.to(device))
        total_loss = loss_reg + loss_cls
        total_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            v_out = model(p_va.to(device), t_va.to(device))
            v_probs = v_out['progression_prob'].cpu().numpy()
            v_preds = (v_probs >= 0.5).astype(int)
            v_ctdna = v_out['ctdna_forecast'].cpu().numpy()

        v_cls = compute_cls_metrics(y_prog_va, v_preds, v_probs)
        if v_cls['f1'] > best_val_f1:
            best_val_f1 = v_cls['f1']
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Test evaluation
    model.eval()
    with torch.no_grad():
        v_out = model(p_va.to(device), t_va.to(device))
        v_probs = v_out['progression_prob'].cpu().numpy()
        v_preds = (v_probs >= 0.5).astype(int)
        v_ctdna = v_out['ctdna_forecast'].cpu().numpy()
    v_cls = compute_cls_metrics(y_prog_va, v_preds, v_probs)
    v_reg = compute_reg_metrics(y_ctdna_va, v_ctdna)
    v_metrics = {**v_cls, **v_reg}

    with torch.no_grad():
        t_out = model(p_te.to(device), t_te.to(device))
        t_probs = t_out['progression_prob'].cpu().numpy()
        t_preds = (t_probs >= 0.5).astype(int)
        t_ctdna = t_out['ctdna_forecast'].cpu().numpy()
    t_cls = compute_cls_metrics(y_prog_te, t_preds, t_probs)
    t_reg = compute_reg_metrics(y_ctdna_te, t_ctdna)
    t_metrics = {**t_cls, **t_reg}

    return model, v_metrics, t_metrics


# =========================================================================
# 5. Master Pipeline Orchestration
# =========================================================================

def run_all_benchmarks(smoke_test=False):
    from dl.training.validated_experiments import main
    return main(smoke_test=smoke_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    run_all_benchmarks(smoke_test=args.smoke_test)
