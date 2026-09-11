"""Confidence calibration and high-confidence error metrics."""
import numpy as np
from typing import List, Dict, Any


def compute_confidence_spread(confidences: List[float]) -> float:
    if len(confidences) < 2:
        return 0.0
    return float(round(max(confidences) - min(confidences), 4))


def is_high_confidence_error(
    actual: Any,
    expected: Any,
    confidence: float,
    threshold: float = 0.85
) -> bool:
    is_wrong = (actual != expected)
    is_high_conf = (confidence >= threshold)
    return bool(is_wrong and is_high_conf)


def compute_confidence_by_outcome(
    results: List[Dict[str, Any]]
) -> Dict[str, float]:
    correct_confs = [r["confidence"] for r in results if r.get("status") == "PASS" and "confidence" in r]
    wrong_confs = [r["confidence"] for r in results if r.get("status") == "FAIL" and "confidence" in r]
    return {
        "mean_correct_confidence": float(round(np.mean(correct_confs), 4)) if correct_confs else 0.0,
        "mean_incorrect_confidence": float(round(np.mean(wrong_confs), 4)) if wrong_confs else 0.0,
        "high_confidence_error_count": sum(1 for r in results if is_high_confidence_error(r.get("actual"), r.get("expected"), r.get("confidence", 0.0)))
    }
