"""
Stage 2 Deep Learning - Dedicated Patient Risk Interpretation Engine

Translates quantitative model predictions into structured research risk categories:
  - LOW:       p < 0.35
  - MODERATE:  0.35 <= p < 0.70
  - HIGH:      p >= 0.70
  - INSUFFICIENT_DATA: missing observations

Enforces mandatory regulatory safeguards:
  - Explicitly labels outputs as "Model-estimated research risk"
  - Never claims clinical diagnosis, staging, or treatment recommendations
"""
from typing import Optional
try:
    from . import config, schemas
except (ImportError, ValueError):
    import config, schemas


class RiskEngine:
    """Interprets model probabilities into research alert categories."""

    LOW_THRESHOLD = 0.35
    HIGH_THRESHOLD = 0.70

    @classmethod
    def stratify_risk(
        cls,
        probability: Optional[float],
        uncertainty: Optional[float] = None
    ) -> schemas.RiskOutput:
        """
        Determines the research risk stratum and joint confidence status.

        Args:
            probability: Model progression probability in [0, 1].
            uncertainty: Epistemic uncertainty score in [0, 1].

        Returns:
            RiskOutput schema with level, joint status, and description.
        """
        if probability is None:
            return schemas.RiskOutput(
                level="INSUFFICIENT_DATA",
                joint_confidence_status="NOT_EVALUATED",
                description="Insufficient data to compute progression risk. Model safely abstained."
            )

        # 1. Stratify Risk Level
        if probability < cls.LOW_THRESHOLD:
            level = "LOW"
        elif probability < cls.HIGH_THRESHOLD:
            level = "MODERATE"
        else:
            level = "HIGH"

        # 2. Joint Confidence / Uncertainty Status
        if uncertainty is not None:
            if level == "HIGH" and uncertainty < 0.20:
                joint_status = "HIGH_RISK_HIGH_CONFIDENCE"
            elif level == "HIGH" and uncertainty >= 0.20:
                joint_status = "HIGH_RISK_LOW_CONFIDENCE"
            elif level == "LOW" and uncertainty < 0.20:
                joint_status = "LOW_RISK_HIGH_CONFIDENCE"
            elif level == "LOW" and uncertainty >= 0.20:
                joint_status = "LOW_RISK_LOW_CONFIDENCE"
            else:
                joint_status = "INTERMEDIATE"
        else:
            joint_status = "INTERMEDIATE"

        description = (
            f"Model-estimated research progression risk: {level} (probability={probability:.4f}). "
            "SYNTHETIC RESEARCH PROTOTYPE ONLY — NOT FOR CLINICAL DIAGNOSIS OR TREATMENT DECISION."
        )

        return schemas.RiskOutput(
            level=level,
            joint_confidence_status=joint_status,
            description=description
        )
