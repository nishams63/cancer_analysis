"""
Stage 2 Deep Learning - Pathology Inference & Attention-MIL Service

Responsible for:
  - Preprocessing biopsy tile images (RGB normalization, resizing)
  - Tile feature extraction (ResNet-50 upgraded, ResNet-18 baseline)
  - Multiple Instance Learning (Attention-MIL pooling vs mean/median/max)
  - Attention weight extraction for explainability
  - Patient-level 128-dimensional pathology representation
"""
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import torch
import torch.nn as nn
from PIL import Image

try:
    from . import config, model_registry, schemas
    from .model_registry import ModelRegistry, MissingCheckpointError
except (ImportError, ValueError):
    import config, model_registry, schemas
    from model_registry import ModelRegistry, MissingCheckpointError

# Import DL model architectures
from dl.models.research_models import PretrainedPathology, CachedMIL
from dl.src.image_model import PathologyCNN
from dl import config as dl_config


class PathologyService:
    """Singleton service for pathology tile inference and aggregation."""
    _instance = None

    def __init__(self):
        self.models: Dict[str, nn.Module] = {}
        self.mil_models: Dict[str, nn.Module] = {}

    @classmethod
    def get_shared(cls) -> 'PathologyService':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_pathology_model(self, backbone: str = 'resnet50') -> nn.Module:
        """Loads and caches the requested pathology backbone in eval mode."""
        if backbone in self.models:
            return self.models[backbone]

        if backbone == 'resnet50':
            ckpt_path = ModelRegistry.verify_checkpoint("pathology_resnet50")
            model = PretrainedPathology('resnet50', load_weights=False)
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            self.models['resnet50'] = model
            return model
        elif backbone == 'resnet18':
            # Check for validated resnet18 first, then baseline
            if ModelRegistry.is_available("pathology_resnet18"):
                ckpt_path = ModelRegistry.get_path("pathology_resnet18")
                model = PretrainedPathology('resnet18', load_weights=False)
                checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                model.load_state_dict(checkpoint['state_dict'])
            elif ModelRegistry.is_available("baseline_pathology"):
                ckpt_path = ModelRegistry.get_path("baseline_pathology")
                model = PathologyCNN(backbone='resnet18', num_classes=3, pretrained=False)
                checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                weights = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
                model.load_state_dict(weights)
            else:
                raise MissingCheckpointError("Neither 'pathology_resnet18' nor 'baseline_pathology' checkpoint exists.")
            model.eval()
            self.models['resnet18'] = model
            return model
        else:
            raise ValueError(f"Unsupported pathology backbone: {backbone}")

    def get_mil_model(self, input_dim: int = 2048, mode: str = 'attention') -> nn.Module:
        """Loads and caches the CachedMIL aggregation head in eval mode."""
        cache_key = f"mil_{input_dim}_{mode}"
        if cache_key in self.mil_models:
            return self.mil_models[cache_key]

        model = CachedMIL(input_dim=input_dim, mode=mode)
        if mode == 'attention' and ModelRegistry.is_available("mil_attention"):
            ckpt_path = ModelRegistry.get_path("mil_attention")
            state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(state.get('state_dict', state))

        model.eval()
        self.mil_models[cache_key] = model
        return model

    def process_tiles(
        self,
        tile_inputs: List[Union[str, Path, Image.Image]],
        backbone: str = 'resnet50',
        aggregation_method: str = 'attention'
    ) -> Tuple[schemas.PathologyOutput, torch.Tensor]:
        """
        Executes tile preprocessing, backbone feature extraction, and MIL aggregation.

        Args:
            tile_inputs: List of tile file paths or PIL Images.
            backbone: 'resnet50' or 'resnet18'.
            aggregation_method: 'attention', 'mean', 'median', or 'max'.

        Returns:
            Tuple of (PathologyOutput, patient_embedding [1, 128])
        """
        if not tile_inputs:
            raise ValueError("tile_inputs list cannot be empty for pathology processing.")

        model = self.get_pathology_model(backbone)
        
        # Load and preprocess all tiles
        processed_tensors = []
        for item in tile_inputs:
            if isinstance(item, (str, Path)):
                with Image.open(str(item)) as img:
                    im = img.convert('RGB')
                    processed_tensors.append(model.preprocess(im) if hasattr(model, 'preprocess') else model.transforms(im))
            elif isinstance(item, Image.Image):
                im = item.convert('RGB')
                processed_tensors.append(model.preprocess(im) if hasattr(model, 'preprocess') else model.transforms(im))
            else:
                raise TypeError(f"Invalid tile type: {type(item)}")

        batch_tiles = torch.stack(processed_tensors)  # (K, 3, 224, 224)

        with torch.no_grad():
            if hasattr(model, 'raw_features'):
                raw_features = model.raw_features(batch_tiles)  # (K, 2048) or (K, 512)
            else:
                raw_features = model.extract_features(batch_tiles)

            raw_dim = raw_features.shape[-1]

            if aggregation_method == 'attention':
                mil = self.get_mil_model(input_dim=raw_dim, mode='attention')
                # CachedMIL expects (B, K, raw_dim)
                logits, patient_repr, weights = mil(raw_features.unsqueeze(0))  # logits: (1, 3), patient_repr: (1, 128), weights: (1, K)
                probs = torch.softmax(logits, dim=-1)[0]
                tile_weights = weights[0].cpu().numpy()

                # Build ranked attention information
                ranked_indices = tile_weights.argsort()[::-1]
                attention_tiles = []
                for rank, idx in enumerate(ranked_indices):
                    tile_path_str = str(tile_inputs[idx]) if isinstance(tile_inputs[idx], (str, Path)) else f"tile_{idx}"
                    attention_tiles.append(schemas.TileAttentionInfo(
                        tile_index=int(idx),
                        tile_path=tile_path_str,
                        attention_weight=float(tile_weights[idx]),
                        rank=int(rank + 1)
                    ))
            else:
                mil = self.get_mil_model(input_dim=raw_dim, mode=aggregation_method)
                logits, patient_repr, weights = mil(raw_features.unsqueeze(0))
                probs = torch.softmax(logits, dim=-1)[0]
                attention_tiles = None

        p_benign = float(probs[0].item())
        p_malignant = float(probs[1].item())
        p_inflammation = float(probs[2].item())

        summary = schemas.PathologyOutput(
            available=True,
            model_name=f"{backbone}_{aggregation_method}",
            malignant_probability=p_malignant,
            benign_probability=p_benign,
            inflammation_probability=p_inflammation,
            aggregation_method=aggregation_method,
            num_tiles_analyzed=len(tile_inputs),
            attention_tiles=attention_tiles,
            embedding_dimension=int(patient_repr.shape[-1])
        )

        return summary, patient_repr
