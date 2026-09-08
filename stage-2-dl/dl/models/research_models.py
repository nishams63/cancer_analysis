"""Controlled pretrained transfer heads; reusable existing MIL/fusion/temporal blocks."""
import torch
from torch import nn
from torchvision import models
from .attention_mil import GatedAttentionMIL


class PretrainedPathology(nn.Module):
    """Official weights required for training; load_weights=False only for checkpoint restore.

    All four architectures receive the same 128-unit head and optimization budget.
    Frozen feature extraction also freezes BatchNorm statistics and backbone dropout.
    """
    SPECS = {
        'resnet18': (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
        'resnet50': (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
        'efficientnet_b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, 1280),
        'vit': (models.vit_b_16, models.ViT_B_16_Weights.DEFAULT, 768),
    }
    def __init__(self, name, load_weights=True):
        super().__init__()
        constructor, weights, dim = self.SPECS[name]
        self.name, self.weights_id, self.weights_url = name, str(weights), weights.url
        self.preprocess = weights.transforms()
        self.encoder = constructor(weights=weights if load_weights else None, progress=False)
        if name.startswith('resnet'):
            self.encoder.fc = nn.Identity()
        elif name == 'vit':
            self.encoder.heads = nn.Identity()
        else:
            self.encoder.classifier = nn.Identity()
        self.encoder.requires_grad_(False)
        self.head = nn.Sequential(nn.Linear(dim,128), nn.ReLU(), nn.Dropout(.3), nn.Linear(128,3))
        self.raw_dim = dim
        self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def raw_features(self, images):
        return self.encoder(images)

    def extract_features(self, images):
        return self.head[:2](self.raw_features(images))

    def forward(self, images):
        return self.head(self.raw_features(images))


class CachedMIL(nn.Module):
    def __init__(self, input_dim, mode='attention'):
        super().__init__()
        self.mode = mode
        self.projection = nn.Sequential(nn.Linear(input_dim,128),nn.ReLU())
        self.attention = GatedAttentionMIL(128,64) if mode == 'attention' else None
        self.classifier = nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Dropout(.25),nn.Linear(64,3))

    def forward(self, x):
        h = self.projection(x)
        if self.mode == 'attention':
            rep, weights = self.attention(h)
        else:
            if self.mode == 'mean':
                rep = h.mean(1)
            elif self.mode == 'median':
                rep = h.median(1).values
            elif self.mode == 'max':
                rep = h.max(1).values
            else:
                raise ValueError(self.mode)
            # Uniform reference weights are not claimed as max/median attributions.
            weights = torch.full(h.shape[:2],1/h.shape[1],device=h.device)
        return self.classifier(rep), rep, weights
