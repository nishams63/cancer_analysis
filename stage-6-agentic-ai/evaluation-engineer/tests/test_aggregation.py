"""Tests for metrics aggregation across suite runs."""
from metrics.aggregation import aggregate_evaluation_results
from schemas.evaluation import ScenarioEvaluationResult
from schemas.metrics import ProcessMetrics, OutcomeMetrics, SafetyMetrics, ExecutionMetrics


def test_aggregate_suite_results():
    proc = ProcessMetrics(
        workflow_adherence=1.0,
        task_ordering_accuracy=1.0,
        tool_selection_accuracy=1.0,
        tool_precision=1.0,
        tool_recall=1.0,
        knowledge_retrieval_accuracy=1.0,
        knowledge_relevance=1.0,
        branch_accuracy=1.0,
        escalation_precision=1.0,
        escalation_recall=1.0,
        escalation_f1=1.0,
        evidence_grounding=1.0,
        safety_score=1.0,
        reliability_score=1.0,
        weighted_process_score=0.90,
    )
    out = OutcomeMetrics(
        analytical_accuracy=0.95,
        numerical_accuracy=0.95,
        factual_accuracy=0.95,
        root_cause_accuracy=0.95,
        recommendation_relevance=0.95,
        recommendation_actionability=0.95,
        recommendation_grounding=0.95,
        overall_outcome_score=0.95,
    )
    safe = SafetyMetrics(
        critical_violations=0,
        high_violations=0,
        medium_violations=0,
        low_violations=0,
        missed_escalations=0,
        hallucinated_facts=0,
        unauthorized_tools=0,
        has_critical_failure=False,
    )
    exec_m = ExecutionMetrics(
        duration_seconds=1.2,
        llm_call_count=3,
        tool_call_count=3,
        retry_count=0,
        tokens_used=500,
        cost_usd=0.005,
    )

    r1 = ScenarioEvaluationResult(
        scenario_id="SC-001",
        run_id="R-001",
        passed=True,
        overall_score=0.92,
        process_score=0.90,
        outcome_score=0.95,
        process_metrics=proc,
        outcome_metrics=out,
        safety_metrics=safe,
        execution_metrics=exec_m,
    )
    r2 = ScenarioEvaluationResult(
        scenario_id="SC-002",
        run_id="R-002",
        passed=False,
        overall_score=0.55,
        process_score=0.60,
        outcome_score=0.50,
        process_metrics=proc,
        outcome_metrics=out,
        safety_metrics=safe,
        execution_metrics=exec_m,
    )
    agg = aggregate_evaluation_results(
        evaluation_id="EVAL-001",
        results=[r1, r2],
        timestamp="2026-09-13T12:00:00Z",
        duration_seconds=3.2,
    )
    assert agg.total_scenarios == 2
    assert agg.passed_scenarios == 1
    assert agg.failed_scenarios == 1
    assert agg.success_rate == 0.50
    assert agg.average_overall_score == 0.735
    assert agg.average_process_score == 0.75
    assert agg.average_outcome_score == 0.725
