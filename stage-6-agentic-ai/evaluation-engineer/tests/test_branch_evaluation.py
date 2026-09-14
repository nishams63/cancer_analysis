"""Tests for BranchEvaluator."""
from evaluators.branch_evaluator import BranchEvaluator
from schemas.trace import TraceEvent, EventType


def test_branch_accuracy_success(sample_scenario, sample_trace_events):
    evaluator = BranchEvaluator()
    acc, fails = evaluator.evaluate(sample_scenario, sample_trace_events)
    assert acc == 1.0
    assert len(fails) == 0


def test_wrong_branch_routing(sample_scenario):
    wrong_branch_trace = [
        TraceEvent(trace_id="T1", run_id="R1", timestamp="now", event_type=EventType.BRANCH_DECISION, next_task_id="T004_WRONG")
    ]
    evaluator = BranchEvaluator()
    acc, fails = evaluator.evaluate(sample_scenario, wrong_branch_trace)
    assert acc == 0.0
    assert len(fails) >= 1
    assert fails[0].category.value == "F08_WRONG_BRANCH"
