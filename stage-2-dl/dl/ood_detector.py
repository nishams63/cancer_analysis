"""
Stage 2 Deep Learning - Out-of-Distribution (OOD) & Distribution Shift Detector
Uses Mahalanobis and feature centroid distance over learned latent representations.
Thresholds are fitted strictly on validation/calibration data.
Outputs:
  - IN-DISTRIBUTION
  - POTENTIAL DISTRIBUTION SHIFT
"""
import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional, Union


class LatentOODDetector:
    """
    Mahalanobis and Centroid Distance OOD Detector on latent representations.
    Learns reference distribution parameters strictly on training and validation partitions.
    """
    def __init__(self, regularize_eps: float = 1e-4):
        self.regularize_eps = regularize_eps
        self.centroid = None
        self.inv_cov = None
        self.threshold = None
        self.fitted = False

    def fit_reference(self, train_embeddings: Union[torch.Tensor, np.ndarray]):
        """
        Fits reference centroid and regularized inverse covariance matrix on in-distribution training data.
        """
        if isinstance(train_embeddings, torch.Tensor):
            X = train_embeddings.detach().cpu().numpy()
        else:
            X = np.array(train_embeddings)

        self.centroid = np.mean(X, axis=0)  # (D,)
        cov = np.cov(X, rowvar=False)        # (D, D)
        
        # Add diagonal shrinkage for positive-definiteness
        d = cov.shape[0] if cov.ndim > 1 else 1
        if d == 1:
            cov = np.array([[cov]])
            self.inv_cov = 1.0 / (cov + self.regularize_eps)
        else:
            reg_cov = cov + (np.eye(d) * self.regularize_eps)
            self.inv_cov = np.linalg.pinv(reg_cov)

        self.fitted = True

    def compute_mahalanobis_distance(self, embeddings: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Computes Mahalanobis distance from reference centroid for input embeddings."""
        assert self.fitted, "OOD Detector must be fitted on reference data first."
        if isinstance(embeddings, torch.Tensor):
            X = embeddings.detach().cpu().numpy()
        else:
            X = np.array(embeddings)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        diff = X - self.centroid  # (N, D)
        # Mahalanobis: sqrt( diff @ inv_cov @ diff.T )
        left = np.dot(diff, self.inv_cov)  # (N, D)
        dist_sq = np.sum(left * diff, axis=1)
        return np.sqrt(np.maximum(dist_sq, 0.0))

    def fit_threshold_on_validation(
        self,
        val_embeddings: Union[torch.Tensor, np.ndarray],
        percentile: float = 95.0
    ) -> float:
        """
        Sets OOD detection threshold at specified percentile of validation set distances.
        Guarantees that test set is strictly untouched during threshold selection.
        """
        val_dists = self.compute_mahalanobis_distance(val_embeddings)
        self.threshold = float(np.percentile(val_dists, percentile))
        return self.threshold

    def predict(self, embeddings: Union[torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """
        Evaluates test or challenge samples.
        Outputs:
          - distances: array of Mahalanobis distances
          - status: List of 'IN-DISTRIBUTION' or 'POTENTIAL DISTRIBUTION SHIFT'
          - ood_flags: boolean array (True if OOD)
          - shift_ratio: proportion of samples flagged as shifted
        """
        assert self.threshold is not None, "Threshold must be fitted on validation data before evaluation."
        dists = self.compute_mahalanobis_distance(embeddings)
        ood_flags = dists > self.threshold
        status = [
            "POTENTIAL DISTRIBUTION SHIFT" if flag else "IN-DISTRIBUTION"
            for flag in ood_flags
        ]

        return {
            'mahalanobis_distances': dists,
            'ood_threshold': self.threshold,
            'ood_flags': ood_flags,
            'status': status,
            'shift_fraction': float(np.mean(ood_flags))
        }
