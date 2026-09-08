"""
Stage 2 Deep Learning - Patient-Level Attention Multiple Instance Learning (Attention-MIL)
Aggregates 12 biopsy tiles per patient into a single patient-level morphological prediction.
Supports comparison across:
  - Mean pooling (Baseline)
  - Median pooling
  - Max pooling
  - Gated Attention pooling (Learned Attention-MIL)
Saves tile attention weights for clinical explainability.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

from .. import config
from . import pathology_benchmarks as pb


class GatedAttentionMIL(nn.Module):
    """
    Ilse et al. (2018) Gated Attention Mechanism for Multiple Instance Learning.
    a_k = softmax( w^T (tanh(V h_k) * sigmoid(U h_k)) )
    """
    def __init__(self, in_features: int = 128, attention_dim: int = 64):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(in_features, attention_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(in_features, attention_dim),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(attention_dim, 1)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Tile embeddings (batch_size, num_tiles, in_features)
        Returns:
            patient_repr: (batch_size, in_features)
            attention_weights: (batch_size, num_tiles)
        """
        v = self.attention_V(h)  # (B, K, att_dim)
        u = self.attention_U(h)  # (B, K, att_dim)
        scores = self.attention_w(v * u).squeeze(-1)  # (B, K)
        weights = F.softmax(scores, dim=1)            # (B, K)
        
        # Weighted sum: (B, 1, K) x (B, K, D) -> (B, 1, D) -> (B, D)
        patient_repr = torch.bmm(weights.unsqueeze(1), h).squeeze(1)
        return patient_repr, weights


class PatientPathologyMIL(nn.Module):
    """
    End-to-end or embedding-based Patient-Level Pathology Architecture.
    Processes K=12 tiles per patient and pools them into a patient representation.
    """
    def __init__(
        self,
        backbone_name: str = 'resnet18',
        num_classes: int = config.NUM_CLASSES,
        feature_dim: int = 128,
        pooling_mode: str = 'attention',
        pretrained: bool = False
    ):
        super().__init__()
        assert pooling_mode in ['mean', 'median', 'max', 'attention'], f"Invalid pooling mode: {pooling_mode}"
        self.pooling_mode = pooling_mode
        self.feature_dim = feature_dim

        # Backbone tile feature extractor
        self.backbone = pb.build_pathology_benchmark_model(backbone_name, pretrained=pretrained, freeze_backbone=True)
        # Match backbone feature dim to feature_dim if different
        self.backbone.eval()
        sample_in = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            sample_feat = self.backbone.extract_features(sample_in)
            actual_in_dim = sample_feat.shape[1]

        if actual_in_dim != feature_dim:
            self.proj = nn.Linear(actual_in_dim, feature_dim)
        else:
            self.proj = nn.Identity()

        # Attention mechanism
        if pooling_mode == 'attention':
            self.attention = GatedAttentionMIL(in_features=feature_dim, attention_dim=64)
        else:
            self.attention = None

        # Patient-level classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )

    def extract_tile_features(self, bag_images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bag_images: (batch_size, num_tiles, 3, 224, 224)
        Returns:
            embeddings: (batch_size, num_tiles, feature_dim)
        """
        B, K, C, H, W = bag_images.shape
        flat_images = bag_images.view(B * K, C, H, W)
        flat_feats = self.backbone.extract_features(flat_images)
        flat_feats = self.proj(flat_feats)
        return flat_feats.view(B, K, self.feature_dim)

    def forward(
        self,
        bag_images: Optional[torch.Tensor] = None,
        tile_embeddings: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Accepts either raw tile bag images or pre-extracted tile embeddings.
        Returns:
            patient_logits: (batch_size, num_classes)
            patient_representation: (batch_size, feature_dim)
            attention_weights: (batch_size, num_tiles)
        """
        if tile_embeddings is None:
            if bag_images is None:
                raise ValueError("Either bag_images or tile_embeddings must be provided")
            h = self.extract_tile_features(bag_images)
        else:
            h = self.proj(tile_embeddings) if not isinstance(self.proj, nn.Identity) else tile_embeddings

        B, K, D = h.shape

        if self.pooling_mode == 'attention':
            patient_repr, weights = self.attention(h)
        elif self.pooling_mode == 'mean':
            patient_repr = torch.mean(h, dim=1)
            weights = torch.full((B, K), 1.0 / K, device=h.device)
        elif self.pooling_mode == 'median':
            patient_repr = torch.median(h, dim=1).values
            weights = torch.full((B, K), 1.0 / K, device=h.device)
        elif self.pooling_mode == 'max':
            patient_repr = torch.max(h, dim=1).values
            weights = torch.full((B, K), 1.0 / K, device=h.device)

        logits = self.classifier(patient_repr)
        return logits, patient_repr, weights
