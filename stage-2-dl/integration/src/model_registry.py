"""
Stage 2 Deep Learning - Centralized Model Checkpoint Registry

Centralizes all model checkpoint paths, verification, and loading policies.
Enforces strict anti-silent-fallback rules: missing checkpoints raise explicit errors.
"""
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from . import config
except (ImportError, ValueError):
    import config


class MissingCheckpointError(FileNotFoundError):
    """Raised when a required model checkpoint does not exist. Never fall back silently."""
    pass


class ModelRegistry:
    """Central registry of trained model checkpoints and artifacts."""

    CHECKPOINTS_DIR = config.CHECKPOINTS_DIR
    VALIDATED_V3_DIR = CHECKPOINTS_DIR / 'validated_v3'
    RESULTS_V3_DIR = config.STAGE_2_DIR / 'results' / 'validated_v3'

    REGISTRY: Dict[str, Path] = {
        # Pathology models
        "pathology_resnet50": VALIDATED_V3_DIR / "pathology_resnet50.pt",
        "pathology_resnet18": VALIDATED_V3_DIR / "pathology_resnet18.pt",
        "baseline_pathology": config.FROZEN_IMAGE_CHECKPOINT,  # best_pathology_cnn.pt
        
        # Temporal models
        "temporal_transformer": VALIDATED_V3_DIR / "transformer_seed42.pt",
        "temporal_bilstm": VALIDATED_V3_DIR / "bilstm_seed42.pt",
        "baseline_temporal": config.FROZEN_TEMPORAL_CHECKPOINT,  # best_temporal_lstm.pt
        
        # Aggregation models
        "mil_attention": VALIDATED_V3_DIR / "mil_attention_seed42.pt",
        
        # Multimodal fusion models
        "fusion_gated": VALIDATED_V3_DIR / "fusion_gated_seed42.pt",
        "fusion_concat_mlp": VALIDATED_V3_DIR / "fusion_concat_mlp_seed42.pt",
        "fusion_cross_attention": VALIDATED_V3_DIR / "fusion_cross_attention_seed42.pt",
        "pathology_only": VALIDATED_V3_DIR / "pathology_only_seed42.pt",
        
        # Reliability & safety artifacts
        "calibration": VALIDATED_V3_DIR / "calibration.pt",
        "ood": VALIDATED_V3_DIR / "ood.pt",
        "selection": RESULTS_V3_DIR / "selection.json",
    }

    @classmethod
    def get_path(cls, key: str) -> Path:
        """Returns the registered path for a given model key."""
        if key not in cls.REGISTRY:
            raise KeyError(f"Unknown model key '{key}'. Available keys: {list(cls.REGISTRY.keys())}")
        return cls.REGISTRY[key]

    @classmethod
    def verify_checkpoint(cls, key: str) -> Path:
        """
        Verifies that a checkpoint exists on disk.
        Raises MissingCheckpointError if missing. Never silently falls back.
        """
        path = cls.get_path(key)
        if not path.exists():
            raise MissingCheckpointError(
                f"Required checkpoint for '{key}' was not found at: {path}. "
                f"Silent fallback is forbidden by the integrity protocol."
            )
        return path

    @classmethod
    def is_available(cls, key: str) -> bool:
        """Checks if a registered checkpoint currently exists on disk."""
        if key not in cls.REGISTRY:
            return False
        return cls.REGISTRY[key].exists()

    @classmethod
    def status_summary(cls) -> Dict[str, Dict[str, Any]]:
        """Returns availability status for all registered model checkpoints."""
        summary = {}
        for key, path in cls.REGISTRY.items():
            exists = path.exists()
            size = path.stat().st_size if exists else 0
            summary[key] = {
                "path": str(path),
                "available": exists,
                "size_bytes": size
            }
        return summary
