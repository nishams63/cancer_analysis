"""
Stage 2 Deep Learning (DL) - Pathology Image Model Architectures
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

try:
    from . import config
except (ImportError, ValueError):
    import config

class ResNet18Transfer(nn.Module):
    """
    Transfer Learning architecture based on ResNet-18 for pathology tile classification.
    Replaces 1000-way ImageNet head with a regularized 3-way pathology classification head.
    """
    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        try:
            self.backbone = models.resnet18(weights=weights)
        except Exception:
            # Fallback to uninitialized weights if offline
            self.backbone = models.resnet18(weights=None)
            
        in_features = self.backbone.fc.in_features  # 512
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Custom classification head
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class PathologyCNN(nn.Module):
    """
    Lightweight, modular 4-stage convolutional neural network trained from scratch.
    Designed for fast, efficient CPU evaluation and benchmarking without external weights.
    """
    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1: 224 -> 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Stage 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Stage 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Stage 4: 28 -> 14
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits


def build_image_model(arch: str = 'resnet18', pretrained: bool = True, num_classes: int = config.NUM_CLASSES, freeze_backbone: bool = False) -> nn.Module:
    """Factory function for image model instantiation."""
    if arch.lower() == 'resnet18':
        return ResNet18Transfer(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    elif arch.lower() == 'custom_cnn':
        return PathologyCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}. Choose 'resnet18' or 'custom_cnn'")
