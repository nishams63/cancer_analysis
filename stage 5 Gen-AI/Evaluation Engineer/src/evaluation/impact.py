"""Failure Impact Score Calculator."""
from typing import Dict, Any, List


class FailureImpactCalculator:
    """Calculates FAILURE IMPACT SCORE (0-100)."""

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "critical_output": 0.35,
            "multi_stage_failure": 0.20,
            "high_confidence_error": 0.20,
            "downstream_propagation": 0.15,
            "reproducibility": 0.10
        }

    def calculate_impact(
        self,
        failure_codes: List[str],
        failed_stages: List[str],
        has_high_confidence_error: bool,
        is_reproducible: bool = True
    ) -> Dict[str, Any]:
        # Critical output: F01, F02, F04, F06 affect critical clinical paths
        critical_codes = {"F01", "F02", "F04", "F06", "F09"}
        is_critical = any(fc in critical_codes for fc in failure_codes)
        c_score = 1.0 if is_critical else 0.3

        # Multi-stage failure
        m_score = min(1.0, len(failed_stages) / 2.0)

        # High confidence error
        h_score = 1.0 if has_high_confidence_error else 0.0

        # Downstream propagation (e.g. Stage 3 NLP entity omission leads to Stage 4 SLM wrong drug)
        p_score = 1.0 if ("stage3" in failed_stages and "stage4" in failed_stages) else 0.2

        # Reproducibility
        r_score = 1.0 if is_reproducible else 0.5

        w = self.weights
        total_impact = (
            w.get("critical_output", 0.35) * c_score +
            w.get("multi_stage_failure", 0.20) * m_score +
            w.get("high_confidence_error", 0.20) * h_score +
            w.get("downstream_propagation", 0.15) * p_score +
            w.get("reproducibility", 0.10) * r_score
        ) * 100.0

        total_impact = round(min(100.0, max(0.0, total_impact)), 2)

        if total_impact >= 75.0:
            severity = "CRITICAL"
        elif total_impact >= 50.0:
            severity = "HIGH"
        elif total_impact >= 25.0:
            severity = "MODERATE"
        else:
            severity = "LOW"

        return {
            "failure_impact_score": total_impact,
            "severity_level": severity,
            "components": {
                "critical_output_score": round(c_score * 100, 1),
                "multi_stage_score": round(m_score * 100, 1),
                "high_confidence_error_score": round(h_score * 100, 1),
                "propagation_score": round(p_score * 100, 1),
                "reproducibility_score": round(r_score * 100, 1)
            }
        }
