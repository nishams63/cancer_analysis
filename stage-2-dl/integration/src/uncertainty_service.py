"""
Stage 2 Deep Learning - Uncertainty & Calibration Service

Responsible for:
  - Applying validation-fitted Temperature Scaling
  - Monte Carlo Dropout (T=15 passes) for Epistemic Uncertainty & Predictive Variance
  - Calibration status and confidence metrics
"""
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np

try:
    from . import config, model_registry, schemas
    from .model_registry import ModelRegistry
except (ImportError, ValueError):
    import config, model_registry, schemas
    from model_registry import ModelRegistry

from dl.uncertainty_calibration import MCDropoutEvaluator, TemperatureScaler


class UncertaintyService:
    """Singleton service for epistemic uncertainty and probability calibration."""
    _instance = None

    def __init__(self, num_mc_passes: int = 15):
        self.num_mc_passes = num_mc_passes
        self.mc_evaluator = MCDropoutEvaluator(num_samples=num_mc_passes)
        self.temperature_scaler: Optional[TemperatureScaler] = None
        self._load_calibration()

    @classmethod
    def get_shared(cls) -> 'UncertaintyService':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_calibration(self):
        """Loads validation-fitted temperature scaling parameters if available."""
        if ModelRegistry.is_available("calibration"):
            try:
                ckpt_path = ModelRegistry.get_path("calibration")
                data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                scaler = TemperatureScaler()
                scaler.load_state_dict(data.get('temperature', data))
                scaler.eval()
                self.temperature_scaler = scaler
            except Exception:
                self.temperature_scaler = None

    def calibrate_probability(self, logits: torch.Tensor) -> Tuple[float, str]:
        """
        Applies learned temperature scaling to model logits.

        Returns:
            Tuple of (calibrated_probability, calibration_status_string)
        """
        if self.temperature_scaler is not None and logits is not None:
            with torch.no_grad():
                scaled_logits = self.temperature_scaler(logits)
                prob = float(torch.sigmoid(scaled_logits)[0].item())
                return prob, "VALIDATION_TEMPERATURE_SCALED"
        elif logits is not None:
            prob = float(torch.sigmoid(logits)[0].item())
            return prob, "RAW_SIGMOID_UNSCALED"
        else:
            return 0.5, "NOT_CALIBRATED"

    def estimate_uncertainty(
        self,
        model: nn.Module,
        *inputs: torch.Tensor,
        current_probability: Optional[float] = None
    ) -> schemas.UncertaintyOutput:
        """
        Runs Monte Carlo Dropout to calculate epistemic uncertainty and predictive variance.

        Args:
            model: Neural network module with dropout layers.
            *inputs: Input tensors to forward through model.
            current_probability: Point prediction probability.

        Returns:
            UncertaintyOutput with confidence, variance, and uncertainty score.
        """
        try:
            mc_results = self.mc_evaluator.evaluate_classification(model, *inputs)
            variance = float(np.mean(mc_results['predictive_variance']))
            uncertainty_score = float(np.mean(mc_results['uncertainty_score']))
        except Exception:
            variance = None
            uncertainty_score = None

        prob = current_probability if current_probability is not None else 0.5
        confidence = max(prob, 1.0 - prob)

        status = "VALIDATION_TEMPERATURE_SCALED" if self.temperature_scaler is not None else "NOT_CALIBRATED"

        return schemas.UncertaintyOutput(
            confidence=round(confidence, 4),
            uncertainty_score=round(uncertainty_score, 4) if uncertainty_score is not None else None,
            predictive_variance=round(variance, 6) if variance is not None else None,
            method=f"mc_dropout_{self.num_mc_passes}_passes",
            calibration_status=status,
            confidence_definition="Maximum predicted class probability; not a clinical guarantee"
        )
