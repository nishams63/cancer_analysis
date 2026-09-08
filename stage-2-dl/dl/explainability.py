"""
Stage 2 Deep Learning - Multimodal Explainability Suite
Includes:
  1. Pathology Grad-CAM: Class activation mapping for histological feature localization
  2. Patient-Level Attention-MIL Tile Importance: Ranks tile influence across the 12 patient tiles
  3. Longitudinal Biomarker Permutation Importance: Measures feature drop on model predictions
Scientific Note: Feature attribution indicates mathematical model sensitivity and does NOT prove biological causation.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional, Union

from . import config


class PathologyGradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) for CNN pathology backbones.
    Localizes morphological salient regions corresponding to target class.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or self._find_target_layer()
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _find_target_layer(self) -> nn.Module:
        """Finds the last convolutional layer in the backbone."""
        if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'layer4'):
            return self.model.backbone.layer4[-1]
        elif hasattr(self.model, 'features'):
            # Custom CNN: find last conv2d
            for layer in reversed(self.model.features):
                if isinstance(layer, nn.Conv2d):
                    return layer
        raise ValueError("Could not automatically locate target convolutional layer. Provide target_layer explicitly.")

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generates 2D Grad-CAM heatmap normalized to [0, 1].
        Args:
            input_tensor: (1, 3, 224, 224)
            target_class: Target class index (0: benign, 1: malignant, 2: inflammation)
        Returns:
            cam_map: (224, 224) float numpy array in [0, 1]
        """
        self.model.zero_grad()
        output = self.model(input_tensor.detach().clone().requires_grad_(True))

        if target_class is None:
            target_class = int(torch.argmax(output, dim=1).item())

        score = output[0, target_class]
        score.backward()

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Upsample to input image resolution (224, 224)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-6:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def rank_patient_mil_tiles(
    attention_weights: Union[torch.Tensor, np.ndarray],
    tile_labels: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Ranks patient tiles by learned Attention-MIL importance weight.
    Args:
        attention_weights: (num_tiles,) array summing to 1.0
        tile_labels: Optional list of morphological labels per tile
    Returns:
        List of tile info sorted in descending order of importance.
    """
    if isinstance(attention_weights, torch.Tensor):
        weights = attention_weights.detach().cpu().numpy()
    else:
        weights = np.array(attention_weights)

    ranked_indices = np.argsort(weights)[::-1]
    results = []
    for rank, idx in enumerate(ranked_indices, start=1):
        item = {
            'rank': rank,
            'tile_index': int(idx) + 1,
            'attention_weight': float(round(float(weights[idx]), 4)),
            'percentage': float(round(float(weights[idx]) * 100, 2))
        }
        if tile_labels and idx < len(tile_labels):
            item['tile_label'] = tile_labels[idx]
        results.append(item)
    return results


class TemporalPermutationImportance:
    """
    Permutation Feature Importance for Longitudinal Biomarker Models.
    Measures performance degradation when shuffling individual biomarker channels.
    """
    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or config.ALL_TEMPORAL_FEATURES

    def compute_importance(
        self,
        model: nn.Module,
        dataset,
        device: torch.device = torch.device('cpu'),
        metric_key: str = 'f1'
    ) -> Dict[str, Any]:
        """
        Computes metric drop for each biomarker feature on dataset sequences.
        """
        model.eval()
        # Collect baseline predictions
        all_feats = []
        all_lens = []
        all_y_prog = []

        for i in range(len(dataset)):
            item = dataset[i]
            all_feats.append(item['features'])
            all_lens.append(item['length'])
            all_y_prog.append(item['target_progression'].item())

        feat_tensor = torch.stack(all_feats).to(device)  # (N, L, 13)
        len_tensor = torch.tensor(all_lens, dtype=torch.long).to(device)
        y_true = np.array(all_y_prog)

        # Baseline evaluation
        with torch.no_grad():
            _, base_logits, _ = model(feat_tensor, len_tensor)
            base_probs = torch.sigmoid(base_logits).cpu().numpy()
            base_preds = (base_probs >= 0.5).astype(int)

        from sklearn.metrics import f1_score, roc_auc_score
        base_f1 = f1_score(y_true, base_preds, zero_division=0)

        importance_scores = {}
        for col_idx, col_name in enumerate(self.feature_names):
            # Create a copy and permute column across patients
            permuted_feats = feat_tensor.clone()
            perm_indices = torch.randperm(permuted_feats.size(0))
            permuted_feats[:, :, col_idx] = permuted_feats[perm_indices, :, col_idx]

            with torch.no_grad():
                _, perm_logits, _ = model(permuted_feats, len_tensor)
                perm_probs = torch.sigmoid(perm_logits).cpu().numpy()
                perm_preds = (perm_probs >= 0.5).astype(int)

            perm_f1 = f1_score(y_true, perm_preds, zero_division=0)
            f1_drop = base_f1 - perm_f1
            importance_scores[col_name] = round(float(f1_drop), 4)

        # Sort features by importance descending
        sorted_importance = dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))

        return {
            'baseline_f1': round(float(base_f1), 4),
            'feature_importance_f1_drop': sorted_importance,
            'scientific_disclaimer': (
                "Feature permutation importance measures empirical model sensitivity in synthetic data. "
                "It does not establish biological, physiological, or clinical causation."
            )
        }
