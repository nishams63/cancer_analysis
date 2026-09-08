"""
Stage 2 Deep Learning - Clinical Challenge & Robustness Evaluation Suite
Evaluates models on Clean Locked Test Set, then under controlled Domain Shift Perturbations:
  - Pathology: Brightness, Contrast, Stain Jitter, Gaussian Blur, Gaussian Noise, JPEG Compression
  - Temporal: Missing Visits, Irregular Intervals, Measurement Noise, Missing Biomarkers, Lab Outliers
  - Multimodal: Missing Pathology, Missing Temporal, Partial Pathology Tiles (6 vs 12)
Calculates Absolute Performance Drop and Percentage Degradation.
"""
import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple, Optional

from . import config, dataset_v2
from .models import pathology_benchmarks as pb
from .models import attention_mil as am
from .models import temporal_transformer as tt



def compute_drop(clean_val: float, perturb_val: float, higher_is_better: bool = True) -> Tuple[float, float]:
    """Calculates absolute drop and percentage degradation."""
    if higher_is_better:
        abs_drop = clean_val - perturb_val
        pct_drop = (abs_drop / max(1e-6, clean_val)) * 100.0 if clean_val > 0 else 0.0
    else:  # Lower is better (e.g. MAE, loss)
        abs_drop = perturb_val - clean_val
        pct_drop = (abs_drop / max(1e-6, clean_val)) * 100.0 if clean_val > 0 else 0.0
    return round(float(abs_drop), 4), round(float(pct_drop), 2)


class RobustnessSuite:
    """
    Automated Challenge Robustness Suite for Stage 2 Deep Learning Prototype.
    """
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device

    # ---------------------------------------------------------------------
    # Pathology Robustness
    # ---------------------------------------------------------------------
    def evaluate_pathology_robustness(
        self,
        model: nn.Module,
        batch_size: int = 32
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluates pathology model on clean test and 6 visual perturbations."""
        challenges = [
            'clean',
            'brightness',
            'contrast',
            'stain_color',
            'blur',
            'noise',
            'compression'
        ]
        results = {}
        clean_f1 = None

        model.eval()
        for ch in challenges:
            ch_arg = None if ch == 'clean' else ch
            ds = dataset_v2.PathologyTileDataset(split='test', challenge=ch_arg)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

            all_preds, all_targets = [], []
            with torch.no_grad():
                for imgs, targets, _ in loader:
                    imgs = imgs.to(self.device)
                    outputs = model(imgs)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_targets.extend(targets.numpy())

            from sklearn.metrics import f1_score, accuracy_score
            y_true = np.array(all_targets)
            y_pred = np.array(all_preds)
            f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
            acc = float(accuracy_score(y_true, y_pred))

            if ch == 'clean':
                clean_f1 = f1
                results[ch] = {
                    'condition': 'Clean Locked Test',
                    'macro_f1': round(f1, 4),
                    'accuracy': round(acc, 4),
                    'absolute_drop_f1': 0.0,
                    'percentage_degradation': 0.0
                }
            else:
                abs_drop, pct_drop = compute_drop(clean_f1, f1, higher_is_better=True)
                results[ch] = {
                    'condition': f'Shift: {ch}',
                    'macro_f1': round(f1, 4),
                    'accuracy': round(acc, 4),
                    'absolute_drop_f1': abs_drop,
                    'percentage_degradation': pct_drop
                }
        return results

    # ---------------------------------------------------------------------
    # Temporal Robustness
    # ---------------------------------------------------------------------
    def evaluate_temporal_robustness(
        self,
        model: nn.Module,
        train_norm_params: Dict[str, Tuple[float, float]],
        batch_size: int = 32
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluates temporal model on clean test and 5 clinical longitudinal perturbations."""
        challenges = [
            'clean',
            'missing_visits',
            'irregular_intervals',
            'measurement_noise',
            'missing_biomarkers',
            'outliers'
        ]
        results = {}
        clean_f1 = None
        clean_mae = None

        model.eval()
        for ch in challenges:
            ch_arg = None if ch == 'clean' else ch
            ds = dataset_v2.ContinuousTemporalSequenceDataset(
                split='test',
                norm_params=train_norm_params,
                challenge=ch_arg
            )
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

            all_y_ctdna, all_p_ctdna = [], []
            all_y_prog, all_p_prog = [], []

            with torch.no_grad():
                for batch in loader:
                    feats = batch['features'].to(self.device)
                    lengths = batch['length'].to(self.device)
                    pred_ctdna, pred_logits, _ = model(feats, lengths)

                    all_y_ctdna.extend(batch['target_ctdna'].numpy())
                    all_p_ctdna.extend(pred_ctdna.cpu().numpy())
                    all_y_prog.extend(batch['target_progression'].numpy())
                    all_p_prog.extend(torch.sigmoid(pred_logits).cpu().numpy())

            from sklearn.metrics import f1_score, mean_absolute_error
            y_prog = np.array(all_y_prog)
            p_prog = (np.array(all_p_prog) >= 0.5).astype(int)
            y_ctdna = np.array(all_y_ctdna)
            p_ctdna = np.array(all_p_ctdna)

            f1 = float(f1_score(y_prog, p_prog, zero_division=0))
            mae = float(mean_absolute_error(y_ctdna, p_ctdna))

            if ch == 'clean':
                clean_f1 = f1
                clean_mae = mae
                results[ch] = {
                    'condition': 'Clean Locked Test',
                    'progression_f1': round(f1, 4),
                    'ctdna_mae': round(mae, 4),
                    'f1_absolute_drop': 0.0,
                    'f1_percentage_degradation': 0.0
                }
            else:
                abs_drop, pct_drop = compute_drop(clean_f1, f1, higher_is_better=True)
                results[ch] = {
                    'condition': f'Shift: {ch}',
                    'progression_f1': round(f1, 4),
                    'ctdna_mae': round(mae, 4),
                    'f1_absolute_drop': abs_drop,
                    'f1_percentage_degradation': pct_drop
                }
        return results

    # ---------------------------------------------------------------------
    # Multimodal Robustness & Modality Dropouts
    # ---------------------------------------------------------------------
    def evaluate_multimodal_robustness(
        self,
        fusion_model: nn.Module,
        test_pathology_reprs: torch.Tensor,
        test_temporal_reprs: torch.Tensor,
        test_prog_targets: np.ndarray,
        test_ctdna_targets: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluates learned multimodal model under missing modality conditions:
          - Full Multimodal (Clean Reference)
          - Missing Pathology (Zeroed Pathology Representation)
          - Missing Temporal (Zeroed Temporal Representation)
          - Partial Attenuation (Simulating Partial Tissue Biopsy or Single Lab Visit)
        """
        scenarios = [
            ('full_multimodal', test_pathology_reprs, test_temporal_reprs),
            ('missing_pathology', torch.zeros_like(test_pathology_reprs), test_temporal_reprs),
            ('missing_temporal', test_pathology_reprs, torch.zeros_like(test_temporal_reprs)),
            ('partial_both', test_pathology_reprs * 0.5, test_temporal_reprs * 0.5)
        ]

        fusion_model.eval()
        from sklearn.metrics import f1_score, roc_auc_score, mean_absolute_error

        results = {}
        clean_f1 = None

        for name, p_rep, t_rep in scenarios:
            with torch.no_grad():
                out = fusion_model(p_rep.to(self.device), t_rep.to(self.device))
                probs = out['progression_prob'].cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                ctdna_pred = out['ctdna_forecast'].cpu().numpy()

            f1 = float(f1_score(test_prog_targets, preds, zero_division=0))
            try:
                auc = float(roc_auc_score(test_prog_targets, probs))
            except Exception:
                auc = float('nan')
            mae = float(mean_absolute_error(test_ctdna_targets, ctdna_pred))

            if name == 'full_multimodal':
                clean_f1 = f1
                results[name] = {
                    'scenario': 'Full Multimodal (Reference)',
                    'progression_f1': round(f1, 4),
                    'progression_auc': round(auc, 4),
                    'ctdna_mae': round(mae, 4),
                    'f1_percentage_degradation': 0.0
                }
            else:
                abs_drop, pct_drop = compute_drop(clean_f1, f1, higher_is_better=True)
                results[name] = {
                    'scenario': name.replace('_', ' ').title(),
                    'progression_f1': round(f1, 4),
                    'progression_auc': round(auc, 4),
                    'ctdna_mae': round(mae, 4),
                    'f1_percentage_degradation': pct_drop
                }
        return results
