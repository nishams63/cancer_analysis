"""Counterfactual / What-If evaluation for causal reasoning sensitivity."""
from typing import Dict, Any, List, Optional
from schemas.result import AgentResult


class RobustnessEvaluator:
    """Evaluates whether the agent changes its conclusions when underlying evidence changes."""

    @staticmethod
    def evaluate_counterfactual(
        baseline_result: AgentResult,
        counterfactual_result: AgentResult,
        altered_factor: str,
    ) -> Dict[str, Any]:
        """Verify that altering evidence changes the agent's primary finding appropriately."""
        base_top = baseline_result.findings[0]["cause"] if baseline_result.findings else "None"
        counter_top = counterfactual_result.findings[0]["cause"] if counterfactual_result.findings else "None"

        is_responsive = base_top.lower() != counter_top.lower()

        return {
            "altered_factor": altered_factor,
            "baseline_conclusion": base_top,
            "counterfactual_conclusion": counter_top,
            "is_responsive_to_evidence": is_responsive,
            "robustness_score": 1.0 if is_responsive else 0.0,
            "verdict": "PASSED" if is_responsive else "FAILED: Agent conclusion was insensitive to evidence change",
        }
