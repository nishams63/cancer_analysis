"""Tests for WorkflowEvaluator."""
from evaluators.workflow_evaluator import WorkflowEvaluator
from schemas.result import AgentResult


def test_perfect_workflow_adherence(sample_scenario, sample_trace_events, sample_agent_result):
    evaluator = WorkflowEvaluator()
    adh, order, fails = evaluator.evaluate(sample_scenario, sample_trace_events, sample_agent_result)
    assert adh == 1.0
    assert order == 1.0
    assert len(fails) == 0


def test_missing_workflow_tasks(sample_scenario, sample_trace_events):
    partial_result = AgentResult(
        run_id="RUN-PARTIAL",
        workflow_id="WF-REVENUE-001",
        status="completed",
        objective="Goal",
        completed_tasks=["T001"],  # Missing T002, T003, T005
        confidence=0.8,
        trace_id="TR-1",
    )
    evaluator = WorkflowEvaluator()
    adh, order, fails = evaluator.evaluate(sample_scenario, sample_trace_events, partial_result)
    assert adh == 0.25
    assert len(fails) >= 1
    assert fails[0].category.value == "F02_WRONG_WORKFLOW"
