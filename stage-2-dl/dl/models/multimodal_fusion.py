"""
Stage 2 Deep Learning - Learned Multimodal Fusion Architectures
Integrates Patient Pathology Representations and Longitudinal Biomarker Representations.
Includes:
  1. FixedWeightedFusion (Baseline Reference)
  2. ConcatMLPFusion (Concatenation + Multi-Layer Perceptron)
  3. GatedMultimodalFusion (Learned Sigmoid Modality Gating)
  4. CrossAttentionFusion (Bidirectional Cross-Attention between Modalities)
Outputs:
  - Progression Risk Probability (sigmoid)
  - 30-Day ctDNA VAF Forecast
  - Prototype Risk Score [0, 1]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

from .. import config


class FixedWeightedFusion:
    """
    Frozen Baseline Fusion Model.
    S = 0.35 * P_malignant + 0.40 * P_progression + 0.25 * normalized_ctdna_risk
    """
    def __init__(
        self,
        w_mal: float = config.MULTIMODAL_FUSION_CONFIG.baseline_w_mal,
        w_prog: float = config.MULTIMODAL_FUSION_CONFIG.baseline_w_prog,
        w_vaf: float = config.MULTIMODAL_FUSION_CONFIG.baseline_w_vaf
    ):
        self.w_mal = w_mal
        self.w_prog = w_prog
        self.w_vaf = w_vaf

    def __call__(
        self,
        p_malignant: torch.Tensor,
        p_progression: torch.Tensor,
        ctdna_forecast: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Normalize ctDNA VAF from [0, 10] to [0, 1]
        vaf_norm = torch.clamp(ctdna_forecast / 10.0, 0.0, 1.0)
        risk_score = (self.w_mal * p_malignant) + (self.w_prog * p_progression) + (self.w_vaf * vaf_norm)
        risk_score = torch.clamp(risk_score, 0.0, 1.0)
        return {
            'progression_prob': p_progression,
            'ctdna_forecast': ctdna_forecast,
            'multimodal_risk_score': risk_score,
            'fusion_type': 'fixed_weighted'
        }


class ConcatMLPFusion(nn.Module):
    """
    Learned Concatenation + MLP Fusion.
    [Pathology Embedding, Temporal Embedding] -> Dense MLP -> Multi-Task Outputs.
    """
    def __init__(
        self,
        pathology_dim: int = 128,
        temporal_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        in_dim = pathology_dim + temporal_dim
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU()
        )

        self.progression_head = nn.Linear(32, 1)
        self.ctdna_head = nn.Linear(32, 1)
        self.risk_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        pathology_repr: torch.Tensor,
        temporal_repr: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        fused_in = torch.cat([pathology_repr, temporal_repr], dim=-1)
        h = self.shared_mlp(fused_in)

        prog_logit = self.progression_head(h).squeeze(-1)
        ctdna_pred = self.ctdna_head(h).squeeze(-1)
        risk_score = torch.sigmoid(prog_logit)  # Only expose the supervised risk head.

        return {
            'progression_logits': prog_logit,
            'progression_prob': torch.sigmoid(prog_logit),
            'ctdna_forecast': ctdna_pred,
            'multimodal_risk_score': risk_score,
            'fused_representation': h,
            'fusion_type': 'concat_mlp'
        }


class GatedMultimodalFusion(nn.Module):
    """
    Learned Gated Multimodal Fusion.
    Learns dynamic sigmoid gating weights for each modality conditioned on patient state:
      g_p = sigmoid(W_p * h_p)
      g_t = sigmoid(W_t * h_t)
    Allows the model to downweight unreliable or noisy modalities dynamically.
    """
    def __init__(
        self,
        pathology_dim: int = 128,
        temporal_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        # Projections to common dimension
        self.proj_p = nn.Linear(pathology_dim, hidden_dim)
        self.proj_t = nn.Linear(temporal_dim, hidden_dim)

        # Modality gates
        self.gate_p = nn.Sequential(
            nn.Linear(pathology_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.gate_t = nn.Sequential(
            nn.Linear(temporal_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU()
        )

        self.progression_head = nn.Linear(32, 1)
        self.ctdna_head = nn.Linear(32, 1)
        self.risk_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        pathology_repr: torch.Tensor,
        temporal_repr: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Compute dynamic modality gates
        gp = self.gate_p(pathology_repr)
        gt = self.gate_t(temporal_repr)

        hp = self.proj_p(pathology_repr) * gp
        ht = self.proj_t(temporal_repr) * gt

        fused_in = torch.cat([hp, ht], dim=-1)
        h = self.fusion_mlp(fused_in)

        prog_logit = self.progression_head(h).squeeze(-1)
        ctdna_pred = self.ctdna_head(h).squeeze(-1)
        risk_score = torch.sigmoid(prog_logit)  # Only expose the supervised risk head.

        return {
            'progression_logits': prog_logit,
            'progression_prob': torch.sigmoid(prog_logit),
            'ctdna_forecast': ctdna_pred,
            'multimodal_risk_score': risk_score,
            'fused_representation': h,
            'pathology_gate': gp.mean(dim=-1),
            'temporal_gate': gt.mean(dim=-1),
            'fusion_type': 'gated'
        }


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional Cross-Attention Multimodal Fusion.
    Treats pathology and temporal representations as multimodal tokens and
    learns cross-modal attention maps between morphology and laboratory kinetics.
    """
    def __init__(
        self,
        pathology_dim: int = 128,
        temporal_dim: int = 64,
        common_dim: int = 64,
        nhead: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.proj_p = nn.Linear(pathology_dim, common_dim)
        self.proj_t = nn.Linear(temporal_dim, common_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim=common_dim, num_heads=nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(common_dim)

        self.post_mlp = nn.Sequential(
            nn.Linear(common_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.progression_head = nn.Linear(32, 1)
        self.ctdna_head = nn.Linear(32, 1)
        self.risk_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        pathology_repr: torch.Tensor,
        temporal_repr: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        B = pathology_repr.shape[0]
        # (B, 1, common_dim)
        token_p = self.proj_p(pathology_repr).unsqueeze(1)
        token_t = self.proj_t(temporal_repr).unsqueeze(1)

        tokens = torch.cat([token_p, token_t], dim=1)  # (B, 2, common_dim)
        attn_out, attn_weights = self.cross_attn(tokens, tokens, tokens)
        tokens_refined = self.norm(tokens + attn_out)

        flat_tokens = tokens_refined.view(B, -1)  # (B, 2 * common_dim)
        h = self.post_mlp(flat_tokens)

        prog_logit = self.progression_head(h).squeeze(-1)
        ctdna_pred = self.ctdna_head(h).squeeze(-1)
        risk_score = torch.sigmoid(prog_logit)  # Only expose the supervised risk head.

        return {
            'progression_logits': prog_logit,
            'progression_prob': torch.sigmoid(prog_logit),
            'ctdna_forecast': ctdna_pred,
            'multimodal_risk_score': risk_score,
            'fused_representation': h,
            'cross_attention_weights': attn_weights,
            'fusion_type': 'cross_attention'
        }


def build_fusion_model(fusion_type: str = 'gated', pathology_dim: int = 128, temporal_dim: int = 64) -> nn.Module:
    """Factory function for multimodal fusion architectures."""
    ftype = fusion_type.lower()
    if ftype == 'concat_mlp':
        return ConcatMLPFusion(pathology_dim=pathology_dim, temporal_dim=temporal_dim)
    elif ftype == 'gated':
        return GatedMultimodalFusion(pathology_dim=pathology_dim, temporal_dim=temporal_dim)
    elif ftype in ['attention', 'cross_attention']:
        return CrossAttentionFusion(pathology_dim=pathology_dim, temporal_dim=temporal_dim)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}. Choose concat_mlp, gated, or cross_attention")
