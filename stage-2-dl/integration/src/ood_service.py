"""
Stage 2 Deep Learning - Out-of-Distribution (OOD) Detection Service

Responsible for:
  - Loading reference centroid, inverse covariance, and validation threshold
  - Computing Mahalanobis distance on patient latent representations
  - Flagging potential distribution shift vs in-distribution
"""
from typing import Dict, Any, Tuple, Optional, Union
import torch
import numpy as np

try:
    from . import config, model_registry, schemas
    from .model_registry import ModelRegistry
except (ImportError, ValueError):
    import config, model_registry, schemas
    from model_registry import ModelRegistry

from dl.ood_detector import LatentOODDetector


class OODService:
    """Singleton service for latent Out-of-Distribution detection."""
    _instance = None

    def __init__(self):
        self.detector = LatentOODDetector()
        self._load_reference()

    @classmethod
    def get_shared(cls) -> 'OODService':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_reference(self):
        """Loads fitted reference parameters from checkpoint if available."""
        if ModelRegistry.is_available("ood"):
            try:
                ckpt_path = ModelRegistry.get_path("ood")
                artifact = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                self.detector.centroid = artifact['centroid']
                self.detector.inv_cov = artifact['inv_cov']
                self.detector.threshold = artifact['threshold']
                self.detector.fitted = True
            except Exception:
                self.detector.fitted = False

    def evaluate_representation(
        self,
        representation: Union[torch.Tensor, np.ndarray]
    ) -> schemas.OODOutput:
        """
        Evaluates a patient's latent representation against reference distribution.

        Args:
            representation: (1, D) tensor or array of patient latent features.

        Returns:
            OODOutput schema with distance, threshold, and status.
        """
        if not self.detector.fitted or self.detector.threshold is None:
            return schemas.OODOutput(
                status="NOT_EVALUATED",
                mahalanobis_distance=None,
                threshold=None
            )

        try:
            if isinstance(representation, torch.Tensor):
                rep_np = representation.detach().cpu().numpy()
            else:
                rep_np = np.asarray(representation)

            if rep_np.ndim == 1:
                rep_np = rep_np.reshape(1, -1)

            results = self.detector.predict(rep_np)
            dist = float(results['mahalanobis_distances'][0])
            thresh = float(self.detector.threshold)
            status = "POTENTIAL_DISTRIBUTION_SHIFT" if dist > thresh else "IN_DISTRIBUTION"

            return schemas.OODOutput(
                status=status,
                mahalanobis_distance=round(dist, 4),
                threshold=round(thresh, 4)
            )
        except Exception:
            return schemas.OODOutput(
                status="NOT_EVALUATED",
                mahalanobis_distance=None,
                threshold=None
            )
