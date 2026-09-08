"""
Stage 2 Deep Learning - Multimodal Fusion Service

Unites 128-dimensional pathology representations with 64-dimensional temporal representations.
Supports:
  1. Gated Multimodal Fusion (learned sigmoid gating)
  2. Concat-MLP Fusion (concatenation + multi-layer perceptron)
  3. Cross-Attention Fusion (bidirectional cross-attention)
  4. Fixed-Weighted Fusion (baseline reference)
"""
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn

try:
    from . import config, model_registry, schemas
    from .model_registry import ModelRegistry
except (ImportError, ValueError):
    import config, model_registry, schemas
    from model_registry import ModelRegistry

# Import fusion architectures
from dl.models.multimodal_fusion import build_fusion_model, FixedWeightedFusion


class FusionService:
    """Singleton service for multimodal fusion."""
    _instance = None

    def __init__(self):
        self.models: Dict[str, Any] = {}

    @classmethod
    def get_shared(cls) -> 'FusionService':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_fusion_model(self, fusion_mode: str = 'gated') -> Any:
        """Loads and caches the fusion model in eval mode."""
        mode = fusion_mode.lower()
        if mode in self.models:
            return self.models[mode]

        if mode in ['fixed', 'fixed_weighted']:
            model = FixedWeightedFusion()
            self.models[mode] = model
            return model

        # Learned fusion architectures
        kind = 'gated' if mode in ['gated', 'fusion_gated'] else (
            'concat_mlp' if mode in ['concat_mlp', 'fusion_concat_mlp'] else 'cross_attention'
        )
        model = build_fusion_model(fusion_type=kind, pathology_dim=128, temporal_dim=64)

        # Look for trained checkpoint
        ckpt_key = f"fusion_{kind}"
        if ModelRegistry.is_available(ckpt_key):
            ckpt_path = ModelRegistry.get_path(ckpt_key)
            state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(state.get('state_dict', state))

        model.eval()
        self.models[mode] = model
        return model

    def fuse(
        self,
        pathology_repr: torch.Tensor,
        temporal_repr: torch.Tensor,
        fusion_mode: str = 'gated',
        p_malignant: Optional[float] = None,
        p_progression: Optional[float] = None,
        ctdna_forecast: Optional[float] = None
    ) -> Tuple[schemas.FusionOutput, Dict[str, torch.Tensor]]:
        """
        Executes multimodal fusion over patient representations.

        Args:
            pathology_repr: (1, 128) patient pathology embedding.
            temporal_repr: (1, 64) patient temporal embedding.
            fusion_mode: 'gated', 'concat_mlp', 'cross_attention', or 'fixed_weighted'.
            p_malignant: Float malignant probability (used for fixed weighted baseline).
            p_progression: Float progression probability (used for fixed weighted baseline).
            ctdna_forecast: Float ctDNA forecast (used for fixed weighted baseline).

        Returns:
            Tuple of (FusionOutput, raw_model_dict)
        """
        mode = fusion_mode.lower()
        model = self.get_fusion_model(mode)

        if mode in ['fixed', 'fixed_weighted']:
            p_mal_t = torch.tensor([p_malignant if p_malignant is not None else 0.5])
            p_prog_t = torch.tensor([p_progression if p_progression is not None else 0.5])
            ctdna_t = torch.tensor([ctdna_forecast if ctdna_forecast is not None else 0.0])
            raw_out = model(p_mal_t, p_prog_t, ctdna_t)
            prog_prob = float(raw_out['progression_prob'][0].item())
            ctdna_pred = float(raw_out['ctdna_forecast'][0].item())
            risk_score = float(raw_out['multimodal_risk_score'][0].item())
            fused_dim = 0
        else:
            with torch.no_grad():
                raw_out = model(pathology_repr, temporal_repr)
                prog_prob = float(raw_out['progression_prob'][0].item())
                ctdna_pred = float(raw_out['ctdna_forecast'][0].item())
                risk_score = float(raw_out['multimodal_risk_score'][0].item())
                fused_dim = int(raw_out['fused_representation'].shape[-1])

        summary = schemas.FusionOutput(
            available=True,
            model_name=mode,
            progression_probability=prog_prob,
            ctdna_forecast=ctdna_pred,
            multimodal_risk_score=risk_score,
            fusion_dimension=fused_dim
        )

        return summary, raw_out
