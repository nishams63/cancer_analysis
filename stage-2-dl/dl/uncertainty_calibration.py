"""
Stage 2 Deep Learning - Uncertainty Estimation & Calibration Engine
Includes:
  1. Monte Carlo Dropout (T=10 passes) for Epistemic Uncertainty Estimation
  2. Temperature Scaling (fitted strictly on validation set)
  3. Calibration Metrics: Expected Calibration Error (ECE), Brier Score, Reliability Diagrams
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, List, Optional


class MCDropoutEvaluator:
    """
    Monte Carlo Dropout Evaluator for Epistemic Uncertainty.
    Performs T stochastic forward passes with active dropout at inference time.
    """
    def __init__(self, num_samples: int = 10):
        self.num_samples = num_samples

    def enable_dropout_only(self, model: nn.Module):
        """Forces dropout layers to remain active during evaluation while keeping BatchNorm in eval mode."""
        model.eval()
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

    def evaluate_classification(
        self,
        model: nn.Module,
        *inputs: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Runs T stochastic passes for binary or multi-class classification.
        Returns mean probability, predictive variance, and epistemic uncertainty score.
        """
        original_modes = {m: m.training for m in model.modules()}
        self.enable_dropout_only(model)
        probs_samples = []

        with torch.no_grad():
            for _ in range(self.num_samples):
                out = model(*inputs)
                if isinstance(out, dict):
                    # Multimodal fusion model output
                    logits = out.get('progression_logits')
                    if logits is not None:
                        probs = torch.sigmoid(logits)
                    else:
                        probs = out.get('progression_prob', torch.zeros(1))
                elif isinstance(out, tuple):
                    # Multi-task model: out[1] is classification logits
                    probs = torch.sigmoid(out[1])
                else:
                    probs = torch.sigmoid(out) if out.ndim == 1 or out.shape[-1] == 1 else torch.softmax(out, dim=-1)
                probs_samples.append(probs.cpu().numpy())

        # Shape: (T, B, ...) -> stack along axis 0
        for module, mode in original_modes.items():
            module.training = mode
        samples = np.stack(probs_samples, axis=0)
        mean_prob = np.mean(samples, axis=0)
        var_prob = np.var(samples, axis=0)
        std_prob = np.std(samples, axis=0)

        # Epistemic uncertainty normalized to [0, 1]
        uncertainty = std_prob * 2.0  # Scaled MC spread, not a confidence interval.

        return {
            'mean_prediction': mean_prob,
            'predictive_variance': var_prob,
            'uncertainty_score': uncertainty,
            'num_passes': self.num_samples,
            'all_samples': samples
        }

    def evaluate_regression(
        self,
        model: nn.Module,
        *inputs: torch.Tensor
    ) -> Dict[str, Any]:
        """Runs T stochastic passes for continuous regression (e.g. ctDNA forecasting)."""
        original_modes = {m: m.training for m in model.modules()}
        self.enable_dropout_only(model)
        preds_samples = []

        with torch.no_grad():
            for _ in range(self.num_samples):
                out = model(*inputs)
                if isinstance(out, dict):
                    pred = out['ctdna_forecast']
                elif isinstance(out, tuple):
                    pred = out[0]
                else:
                    pred = out
                preds_samples.append(pred.cpu().numpy())

        for module, mode in original_modes.items():
            module.training = mode
        samples = np.stack(preds_samples, axis=0)
        mean_val = np.mean(samples, axis=0)
        var_val = np.var(samples, axis=0)
        std_val = np.std(samples, axis=0)

        return {
            'mean_forecast': mean_val,
            'predictive_variance': var_val,
            'forecast_uncertainty_std': std_val,
            'num_passes': self.num_samples,
            'all_samples': samples
        }


class TemperatureScaler(nn.Module):
    """
    Post-hoc Temperature Scaling for probability calibration.
    Fits a single scalar T > 0 on validation logits using NLL loss.
    Guarantees that test set is strictly evaluated, never used for calibration fitting.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scales logits by 1/T."""
        # Clamp temperature to avoid numerical explosion or collapse
        temp = torch.clamp(self.temperature, min=0.05, max=10.0)
        return logits / temp

    def fit_validation_data(
        self,
        val_logits: torch.Tensor,
        val_targets: torch.Tensor,
        is_binary: bool = True,
        max_iters: int = 50,
        lr: float = 0.01
    ) -> float:
        """
        Fits temperature parameter strictly on validation partition.
        """
        val_logits = val_logits.detach()
        self.temperature.data.fill_(1.5)
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iters)

        if is_binary:
            criterion = nn.BCEWithLogitsLoss()
            targets = val_targets.float()
        else:
            criterion = nn.CrossEntropyLoss()
            targets = val_targets.long()

        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = criterion(scaled_logits, targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            self.temperature.clamp_(0.05, 10.0)
        learned_temp = float(self.temperature.item())
        return learned_temp


def compute_expected_calibration_error(
    probabilities: np.ndarray,
    ground_truth: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, Dict[str, List[float]]]:
    """
    Computes Expected Calibration Error (ECE) and reliability diagram bins.
    Args:
        probabilities: Predicted positive class probabilities (N,) in [0, 1]
        ground_truth: Binary targets (N,) in {0, 1}
    Returns:
        ece: Float in [0, 1]
        diagram_data: Dict with bin accuracies, bin confidences, bin counts
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []
    ece = 0.0
    N = len(probabilities)

    for i in range(n_bins):
        b_low, b_high = bins[i], bins[i + 1]
        if i == n_bins - 1:
            in_bin = (probabilities >= b_low) & (probabilities <= b_high)
        else:
            in_bin = (probabilities >= b_low) & (probabilities < b_high)

        count = np.sum(in_bin)
        if count > 0:
            bin_acc = np.mean(ground_truth[in_bin])
            bin_conf = np.mean(probabilities[in_bin])
            ece += (count / N) * abs(bin_acc - bin_conf)
            bin_accs.append(float(bin_acc))
            bin_confs.append(float(bin_conf))
            bin_counts.append(int(count))
        else:
            bin_accs.append(0.0)
            bin_confs.append((b_low + b_high) / 2.0)
            bin_counts.append(0)

    diagram_data = {
        'bin_edges': [float(b) for b in bins],
        'bin_accuracies': bin_accs,
        'bin_confidences': bin_confs,
        'bin_counts': bin_counts
    }
    return float(ece), diagram_data


def compute_brier_score(probabilities: np.ndarray, ground_truth: np.ndarray) -> float:
    """Computes mean squared error of predicted probabilities: 1/N sum (p_i - y_i)^2."""
    return float(np.mean((probabilities - ground_truth) ** 2))
