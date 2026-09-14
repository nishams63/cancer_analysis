"""Tests for process, outcome, and aggregation metrics."""
from metrics.process_metrics import compute_weighted_process_score
from metrics.outcome_metrics import compute_composite_outcome_score
from metrics.aggregation import aggregate_evaluation_results


def test_process_score_weighting():
    score = compute_weighted_process_score(
        workflow_adherence=1.0,
        tool_selection_accuracy=1.0,
        knowledge_retrieval_accuracy=1.0,
        branch_accuracy=1.0,
        escalation_recall=1.0,
        evidence_grounding=1.0,
        analytical_accuracy=1.0,
        safety_score=1.0,
        reliability_score=1.0,
    )
    assert score == 1.0


def test_outcome_score_composite():
    score = compute_composite_outcome_score(
        analytical_accuracy=0.90,
        evidence_grounding=0.80,
        recommendation_relevance=0.85,
    )
    expected = (0.40 * 0.90) + (0.30 * 0.80) + (0.30 * 0.85)
    assert score == round(expected, 4)
