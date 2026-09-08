"""
Stage 2 Deep Learning - Continuous Irregular Temporal Transformer
Models longitudinal biomarker sequences with explicit continuous time handling:
  - Continuous time embedding for days_from_baseline and delta_days
  - Multi-head self-attention with boolean key-padding suppression
  - Multi-task output heads (ctDNA 30-day regression + progression classification)
  - Configurable loss weighting (regression + lambda_cls * classification)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

from .. import config


class ContinuousTimeEmbedding(nn.Module):
    """
    Continuous Time Embedding Module for irregular clinical time intervals.
    Encodes both absolute time (days_from_baseline) and relative interval (delta_days)
    using harmonic frequency basis functions projected to d_model.
    """
    def __init__(self, d_model: int = 64, num_freqs: int = 16):
        super().__init__()
        self.d_model = d_model
        self.num_freqs = num_freqs
        # Frequencies spanning hours to hundreds of days
        freqs = torch.exp(torch.linspace(math.log(1e-3), math.log(1.0), num_freqs))
        self.register_buffer('freqs', freqs)
        
        # Linear projection of [sin(wt), cos(wt), delta_t] -> d_model
        in_dim = (num_freqs * 2) + 1
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, days_from_baseline: torch.Tensor, delta_days: torch.Tensor) -> torch.Tensor:
        """
        Args:
            days_from_baseline: (B, L)
            delta_days: (B, L)
        Returns:
            time_emb: (B, L, d_model)
        """
        B, L = days_from_baseline.shape
        t = days_from_baseline.unsqueeze(-1) * self.freqs  # (B, L, num_freqs)
        sin_t = torch.sin(t)
        cos_t = torch.cos(t)
        dt = delta_days.unsqueeze(-1) / 30.0  # Normalized to months
        
        harmonics = torch.cat([sin_t, cos_t, dt], dim=-1)  # (B, L, 2*num_freqs + 1)
        return self.proj(harmonics)


class ContinuousTemporalTransformer(nn.Module):
    """
    Multi-Task Longitudinal Transformer Encoder.
    Processes irregular biomarker sequences up to Day 90.
    Dual Multi-Task Heads:
      - Head A: 30-day forward ctDNA VAF (continuous regression)
      - Head B: Future progression risk (binary classification)
    """
    def __init__(
        self,
        input_dim: int = config.NUM_TEMPORAL_FEATURES,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        max_seq_len: int = 10
    ):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim

        # Input feature projection
        self.feature_proj = nn.Linear(input_dim, d_model)
        # Continuous time representation
        self.time_emb = ContinuousTimeEmbedding(d_model=d_model, num_freqs=16)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Multi-task heads
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        self.classification_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def extract_patient_representation(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Encodes sequence and extracts representation strictly from the last valid historical visit.
        Suppresses padding contamination.
        """
        B, L, D = x.shape
        
        # Extract days_from_baseline (index 7) and delta_days (index 6) for continuous time embedding
        # Note: In config.py: TEMPORAL_NUMERICAL_FEATURES[6] is delta_days, [7] is days_from_baseline
        delta_days = x[:, :, 6]
        days_from_baseline = x[:, :, 7]

        # Key padding mask: True indicates positions to be ignored
        mask = torch.arange(L, device=x.device).expand(B, L) >= lengths.unsqueeze(1)

        feat_h = self.feature_proj(x)
        time_h = self.time_emb(days_from_baseline, delta_days)
        h = self.drop(self.layer_norm(feat_h + time_h))

        encoded = self.transformer(h, src_key_padding_mask=mask)

        # Extract representation strictly at each patient's last valid historical timepoint (lengths[i] - 1)
        last_indices = (lengths - 1).clamp(min=0).unsqueeze(1).unsqueeze(2).expand(-1, 1, self.d_model)
        patient_repr = encoded.gather(1, last_indices).squeeze(1)  # (B, d_model)
        return patient_repr

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, max_seq_len, input_dim)
            lengths: (batch_size,) valid visits
        Returns:
            ctdna_pred: (batch_size,) predicted 30-day future ctDNA VAF
            prog_logits: (batch_size,) predicted progression logits
            patient_repr: (batch_size, d_model) latent temporal state
        """
        patient_repr = self.extract_patient_representation(x, lengths)
        ctdna_pred = self.regression_head(patient_repr).squeeze(-1)
        prog_logits = self.classification_head(patient_repr).squeeze(-1)
        return ctdna_pred, prog_logits, patient_repr

    @staticmethod
    def compute_multi_task_loss(
        ctdna_pred: torch.Tensor,
        ctdna_target: torch.Tensor,
        prog_logits: torch.Tensor,
        prog_target: torch.Tensor,
        lambda_cls: float = 1.0,
        reg_loss_type: str = 'smooth_l1'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes weighted multi-task loss.
        Supports SmoothL1 or MAE regression loss.
        """
        if reg_loss_type == 'mae':
            reg_loss = F.l1_loss(ctdna_pred, ctdna_target)
        else:
            reg_loss = F.smooth_l1_loss(ctdna_pred, ctdna_target)

        cls_loss = F.binary_cross_entropy_with_logits(prog_logits, prog_target)
        total_loss = reg_loss + (lambda_cls * cls_loss)
        return total_loss, reg_loss, cls_loss
