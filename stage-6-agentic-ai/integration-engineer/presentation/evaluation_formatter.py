"""Formats EvaluationResult into clean quality metrics scorecard."""
from typing import Optional, Dict, Any
from evaluation_engineer.schemas.evaluation import ScenarioEvaluationResult


class EvaluationFormatter:
    """Transforms ScenarioEvaluationResult into human-friendly dashboard data."""

    @staticmethod
    def format(eval_result: Optional[ScenarioEvaluationResult]) -> Dict[str, Any]:
        if not eval_result:
            return {
                "evaluated": False,
                "passed": False,
                "verdict": "NOT_EVALUATED",
                "process_score": None,
                "outcome_score": None,
                "overall_score": None,
                "process_metrics": {},
                "outcome_metrics": {},
                "safety": {"critical_violations": 0, "high_violations": 0},
                "failures": [],
            }

        failures_summary = [
            {
                "failure_id": f.failure_id,
                "category": f.category.value if hasattr(f.category, "value") else str(f.category),
                "severity": f.severity.value if hasattr(f.severity, "value") else str(f.severity),
                "explanation": f.explanation,
                "fix": f.recommended_fix,
            }
            for f in eval_result.failures
        ]

        return {
            "evaluated": True,
            "passed": eval_result.passed,
            "verdict": eval_result.verdict_rationale or ("PASS" if eval_result.passed else "FAIL"),
            "process_score": eval_result.process_score,
            "outcome_score": eval_result.outcome_score,
            "overall_score": eval_result.overall_score,
            "process_metrics": {
                "workflow_adherence": eval_result.process_metrics.workflow_adherence,
                "tool_selection": eval_result.process_metrics.tool_selection_accuracy,
                "knowledge_retrieval": eval_result.process_metrics.knowledge_retrieval_accuracy,
                "branch_accuracy": eval_result.process_metrics.branch_accuracy,
                "escalation_recall": eval_result.process_metrics.escalation_recall,
                "evidence_grounding": eval_result.process_metrics.evidence_grounding,
            },
            "outcome_metrics": {
                "analytical_accuracy": eval_result.outcome_metrics.analytical_accuracy,
                "numerical_accuracy": eval_result.outcome_metrics.numerical_accuracy,
                "root_cause_accuracy": eval_result.outcome_metrics.root_cause_accuracy,
                "recommendation_relevance": eval_result.outcome_metrics.recommendation_relevance,
            },
            "safety": {
                "critical_violations": eval_result.safety_metrics.critical_violations,
                "high_violations": eval_result.safety_metrics.high_violations,
                "has_critical_failure": eval_result.safety_metrics.has_critical_failure,
            },
            "failures": failures_summary,
        }
