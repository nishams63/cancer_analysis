"""
Stage 2 Deep Learning - Controlled Pathology Model Benchmark Architectures
Includes:
  1. ResNet-18 (Baseline Transfer Architecture)
  2. ResNet-50 (Deeper Residual Network)
  3. EfficientNet-B0 (Compound Scaling Architecture)
  4. Vision Transformer (Pathology ViT with Patch Multi-Head Attention)
"""
import math
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple, Dict, Any

from .. import config


class ResNet18Transfer(nn.Module):
    """
    Baseline ResNet-18 Transfer Learning Architecture.
    Feature dimension: 128.
    Preserved exactly for direct baseline comparison.
    """
    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = False, freeze_backbone: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        # A failed pretrained download is an error, never a random-weight fallback.
        self.backbone = models.resnet18(weights=weights)

        in_features = self.backbone.fc.in_features  # 512
        if freeze_backbone and pretrained:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.feature_extractor = nn.Sequential(
            *list(self.backbone.children())[:-1],
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )
        self.backbone.fc = self.fc

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(x)  # (B, 512)
        return self.fc[0:2](feat)          # (B, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ResNet50Transfer(nn.Module):
    """
    Deeper Residual Network (ResNet-50) for fine-grained cellular patterns.
    Bottleneck blocks provide multi-scale representation.
    Feature dimension: 256.
    """
    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = False, freeze_backbone: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        # A failed pretrained download is an error, never a random-weight fallback.
        self.backbone = models.resnet50(weights=weights)

        in_features = self.backbone.fc.in_features  # 2048
        if freeze_backbone and pretrained:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.feature_extractor = nn.Sequential(
            *list(self.backbone.children())[:-1],
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
        self.backbone.fc = self.head

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(x)
        return self.head[0:3](feat)  # (B, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class EfficientNetB0Transfer(nn.Module):
    """
    EfficientNet-B0 utilizing mobile inverted bottleneck convolutions (MBConv)
    with squeeze-and-excitation optimization.
    Feature dimension: 256.
    """
    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = False, freeze_backbone: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        # A failed pretrained download is an error, never a random-weight fallback.
        self.backbone = models.efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features  # 1280
        if freeze_backbone and pretrained:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.features(x)
        feat = self.backbone.avgpool(feat)
        feat = torch.flatten(feat, 1)
        return self.backbone.classifier[0:3](feat)  # (B, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class PatchEmbedding(nn.Module):
    """Splits 224x224 RGB image into non-overlapping 16x16 patches and projects to d_model."""
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2  # (224/16)^2 = 196
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224) -> proj: (B, embed_dim, 14, 14) -> flatten: (B, embed_dim, 196) -> transpose: (B, 196, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


class PathologyViT(nn.Module):
    """
    Vision Transformer (ViT) for Pathology Tiles.
    Splits image into 196 patches (16x16), prepends learnable CLS token,
    adds 1D learnable position embeddings, and processes through TransformerEncoder.
    Reproducible, lightweight, and executes reliably on CPU/GPU without external dependencies.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = config.NUM_CLASSES,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.2
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.transformer(x)
        cls_rep = self.norm(x[:, 0])
        return self.head[0:2](cls_rep)  # (B, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.transformer(x)
        cls_rep = self.norm(x[:, 0])
        return self.head(cls_rep)


def build_pathology_benchmark_model(model_name: str, pretrained: bool = False, freeze_backbone: bool = True) -> nn.Module:
    """Factory function for instantiating pathology benchmark models."""
    name = model_name.lower()
    if name == 'resnet18':
        return ResNet18Transfer(pretrained=pretrained, freeze_backbone=freeze_backbone)
    elif name == 'resnet50':
        return ResNet50Transfer(pretrained=pretrained, freeze_backbone=freeze_backbone)
    elif name in ['efficientnet', 'efficientnet_b0', 'efficientnet-b0']:
        return EfficientNetB0Transfer(pretrained=pretrained, freeze_backbone=freeze_backbone)
    elif name in ['vit', 'vision_transformer']:
        return PathologyViT()
    else:
        raise ValueError(f"Unknown pathology model: {model_name}. Choose from: resnet18, resnet50, efficientnet_b0, vit")
