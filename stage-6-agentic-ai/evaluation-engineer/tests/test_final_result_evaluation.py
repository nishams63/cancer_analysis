"""Tests for FinalResultEvaluator."""
from evaluators.final_result_evaluator import FinalResultEvaluator
from schemas.result import AgentResult


def test_final_result_within_tolerance(sample_scenario, sample_ground_truth):
    evaluator = FinalResultEvaluator()
    agent_result = AgentResult(
        run_id="RUN-TEST-002",
        workflow_id="WF-REVENUE-001",
        status="completed",
        objective="Determine why revenue decreased",
        findings=[
            {"cause": "Price Increase in Enterprise Tier", "score": 0.85, "details": "decline_rate 0.182"}
        ],
        evidence=[
            {"decline_rate": 0.182, "decline_magnitude": 178000.0}
        ],
        recommendations=[
            {"recommendation": "Rebalance Enterprise pricing tiers."}
        ],
        confidence=0.85,
        escalated=False,
        completed_tasks=["T001", "T002", "T003", "T005"],
        trace_id="TRACE-002",
    )
    outcome_metrics, fails = evaluator.evaluate(agent_result, sample_ground_truth)
    assert outcome_metrics.overall_outcome_score > 0.80
    assert outcome_metrics.numerical_accuracy == 1.0
    assert outcome_metrics.root_cause_accuracy == 1.0
    assert outcome_metrics.recommendation_relevance == 1.0
    assert len(fails) == 0


def test_final_result_out_of_tolerance(sample_scenario, sample_ground_truth):
    evaluator = FinalResultEvaluator()
    agent_result = AgentResult(
        run_id="RUN-TEST-003",
        workflow_id="WF-REVENUE-001",
        status="completed",
        objective="Determine why revenue decreased",
        findings=[
            {"cause": "Server Outage", "score": 0.9, "details": "decline_rate 0.50"}
        ],
        evidence=[
            {"decline_rate": 0.50, "decline_magnitude": 500000.0}
        ],
        recommendations=[],
        confidence=0.9,
        escalated=False,
        completed_tasks=["T001"],
        trace_id="TRACE-003",
    )
    outcome_metrics, fails = evaluator.evaluate(agent_result, sample_ground_truth)
    assert outcome_metrics.overall_outcome_score < 0.50
    assert any(f.category.value == "F13_NUMERICAL_ERROR" for f in fails)
