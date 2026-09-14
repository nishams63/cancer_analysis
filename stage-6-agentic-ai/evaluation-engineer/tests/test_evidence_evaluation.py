"""Tests for EvidenceEvaluator."""
from evaluators.evidence_evaluator import EvidenceEvaluator
from schemas.result import AgentResult
from schemas.ground_truth import GroundTruth


def test_evidence_grounding_success(sample_agent_result, sample_ground_truth):
    evaluator = EvidenceEvaluator()
    score, fails = evaluator.evaluate(sample_agent_result, sample_ground_truth)
    assert score == 1.0
    assert len(fails) == 0


def test_forbidden_claim_hallucination_detection(sample_ground_truth):
    hallucinated_result = AgentResult(
        run_id="R3",
        workflow_id="WF-1",
        status="completed",
        objective="Goal",
        findings=[
            {"cause": "Revenue increased during Q2 and Q3 miraculously"}
        ],
        confidence=0.9,
        trace_id="T3",
    )
    evaluator = EvidenceEvaluator()
    score, fails = evaluator.evaluate(hallucinated_result, sample_ground_truth)
    assert len(fails) >= 1
    assert fails[0].category.value == "F12_HALLUCINATED_FACT"
    assert fails[0].severity.value == "critical"
