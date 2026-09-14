"""Tests for ToolEvaluator."""
from evaluators.tool_evaluator import ToolEvaluator
from schemas.trace import TraceEvent, EventType


def test_tool_evaluation_success(sample_scenario, sample_trace_events, sample_ground_truth):
    evaluator = ToolEvaluator()
    acc, prec, rec, fails = evaluator.evaluate(sample_scenario, sample_trace_events, sample_ground_truth)
    assert prec == 1.0
    assert rec == 1.0
    assert acc == 1.0
    assert len(fails) == 0


def test_unexpected_tool_detection(sample_scenario, sample_ground_truth):
    trace_with_rogue_tool = [
        TraceEvent(trace_id="T1", run_id="R1", timestamp="now", event_type=EventType.TOOL_CALL, tool_name="unauthorized_data_export"),
    ]
    evaluator = ToolEvaluator()
    acc, prec, rec, fails = evaluator.evaluate(sample_scenario, trace_with_rogue_tool, sample_ground_truth)
    assert prec == 0.0
    assert len(fails) >= 1
    assert fails[0].category.value == "F05_WRONG_TOOL"
