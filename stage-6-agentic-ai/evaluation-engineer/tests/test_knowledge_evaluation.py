"""Tests for KnowledgeEvaluator."""
from evaluators.knowledge_evaluator import KnowledgeEvaluator
from schemas.trace import TraceEvent, EventType


def test_knowledge_retrieval_success(sample_scenario, sample_trace_events):
    evaluator = KnowledgeEvaluator()
    acc, rel, fails = evaluator.evaluate(sample_scenario, sample_trace_events)
    assert acc == 1.0
    assert rel == 1.0
    assert len(fails) == 0


def test_missing_knowledge_retrieval(sample_scenario):
    trace_without_kr = [
        TraceEvent(trace_id="T1", run_id="R1", timestamp="now", event_type=EventType.TOOL_CALL, tool_name="load_dataset")
    ]
    evaluator = KnowledgeEvaluator()
    acc, rel, fails = evaluator.evaluate(sample_scenario, trace_without_kr)
    assert acc == 0.0
    assert len(fails) >= 1
    assert fails[0].category.value == "F06_MISSING_KNOWLEDGE_RETRIEVAL"
