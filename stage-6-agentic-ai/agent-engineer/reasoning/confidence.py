"""Deterministic Confidence Derivation System."""
from typing import Optional
from pydantic import BaseModel, Field


class ConfidenceFactors(BaseModel):
    """Constituent evidence factors used to derive overall analytical confidence."""
    data_quality: float = Field(..., ge=0.0, le=1.0)
    statistical_strength: float = Field(..., ge=0.0, le=1.0)
    sample_size_factor: float = Field(..., ge=0.0, le=1.0)
    evidence_separation: float = Field(..., ge=0.0, le=1.0)


def calculate_confidence(
    data_quality_score: float = 0.85,
    p_value: Optional[float] = 0.001,
    sample_size: int = 10000,
    cause_margin: float = 0.15,
    consistency_score: float = 0.85,
    has_unaddressed_critical_anomaly: bool = False,
) -> float:
    """Compute an objective, empirical confidence score (0.0 to 1.0) without LLM fabrication.
    
    Formula:
      35% Data Quality + 25% Statistical Significance + 20% Consistency + 20% Separation Margin
      Penalty: -0.15 if has_unaddressed_critical_anomaly is True.
    """
    dq_score = max(0.0, min(1.0, data_quality_score))

    if p_value is not None:
        stat_score = max(0.0, min(1.0, 1.0 - (p_value * 10.0)))
    else:
        stat_score = 0.70

    consistency = max(0.0, min(1.0, consistency_score))
    margin_score = min(1.0, max(0.0, cause_margin / 0.20))

    total = (0.35 * dq_score) + (0.25 * stat_score) + (0.20 * consistency) + (0.20 * margin_score)
    if has_unaddressed_critical_anomaly:
        total -= 0.15

    return round(max(0.0, min(1.0, total)), 2)
