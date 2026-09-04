"""
Stage 2 Deep Learning (DL) - Temporal Sequence Model Architectures
"""
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional

try:
    from . import config
except (ImportError, ValueError):
    import config

class BiLSTMForecaster(nn.Module):
    """
    Bidirectional LSTM with Multi-Task Prediction Heads.
    Processes historical biomarker trajectories (t <= 90 days).
    Extracts the unpadded latent state corresponding to each patient's true last historical visit.
    Dual Heads:
      - Head A: 30-Day Forward ctDNA VAF (continuous regression)
      - Head B: Future Progression / Recurrence Risk (binary classification logit)
    """
    def __init__(
        self,
        input_dim: int = config.NUM_TEMPORAL_FEATURES,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        bilstm_out_dim = hidden_dim * 2  # Bidirectional

        # Head A: 30-Day Forward ctDNA VAF Regression
        self.regression_head = nn.Sequential(
            nn.Linear(bilstm_out_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        # Head B: Future Progression / Recurrence Binary Classification
        self.classification_head = nn.Sequential(
            nn.Linear(bilstm_out_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Padded sequence tensor (batch_size, max_seq_len, input_dim)
            lengths: True observation count per patient before padding (batch_size,)
        Returns:
            ctdna_pred: Predicted ctDNA VAF at t + 30 days (batch_size,)
            prog_logits: Logits for future progression (batch_size,)
        """
        out, _ = self.lstm(x)  # (batch_size, max_seq_len, hidden_dim * 2)

        batch_size = x.size(0)
        # Extract representation strictly at each patient's last valid historical timepoint (lengths[i] - 1)
        # This prevents padding tokens from contaminating the sequence representation
        last_indices = (lengths - 1).clamp(min=0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1)
        last_indices = last_indices.expand(-1, 1, out.size(2))               # (batch_size, 1, hidden_dim*2)
        patient_repr = out.gather(1, last_indices).squeeze(1)                # (batch_size, hidden_dim*2)

        ctdna_pred = self.regression_head(patient_repr).squeeze(-1)
        prog_logits = self.classification_head(patient_repr).squeeze(-1)

        return ctdna_pred, prog_logits


class TemporalTransformerForecaster(nn.Module):
    """
    Self-Attention Transformer Encoder for longitudinal sequence modeling.
    Uses continuous positional encoding and attention masking for valid steps.
    """
    def __init__(
        self,
        input_dim: int = config.NUM_TEMPORAL_FEATURES,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

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

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_len, _ = x.size()
        # Key padding mask: True indicates position to be ignored
        mask = torch.arange(max_len, device=x.device).expand(batch_size, max_len) >= lengths.unsqueeze(1)

        h = self.input_proj(x)
        encoded = self.transformer(h, src_key_padding_mask=mask)

        # Extract last valid position
        last_indices = (lengths - 1).clamp(min=0).unsqueeze(1).unsqueeze(2).expand(-1, 1, encoded.size(2))
        patient_repr = encoded.gather(1, last_indices).squeeze(1)

        ctdna_pred = self.regression_head(patient_repr).squeeze(-1)
        prog_logits = self.classification_head(patient_repr).squeeze(-1)

        return ctdna_pred, prog_logits


def build_temporal_model(
    arch: str = 'lstm',
    input_dim: int = config.NUM_TEMPORAL_FEATURES,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2
) -> nn.Module:
    """Factory function for temporal model instantiation."""
    if arch.lower() == 'lstm':
        return BiLSTMForecaster(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    elif arch.lower() == 'transformer':
        return TemporalTransformerForecaster(input_dim=input_dim, d_model=hidden_dim, num_layers=num_layers, dropout=dropout)
    else:
        raise ValueError(f"Unknown architecture: {arch}. Choose 'lstm' or 'transformer'")
