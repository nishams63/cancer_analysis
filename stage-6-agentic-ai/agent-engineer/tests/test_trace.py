"""Tests for SQLite trace persistence, event recording, and human override tracking."""
from schemas.agent import AgentState, RunStatus
from schemas.trace import EventType
from trace.recorder import TraceRecorder


def test_trace_recording_and_query(trace_repo):
    recorder = TraceRecorder(repository=trace_repo)
    recorder.record(
        run_id="RUN-TRACE-01",
        task_id="T001",
        event_type=EventType.WORKFLOW_STARTED,
        decision="start",
        decision_reason="Initiating pipeline",
    )
    recorder.record(
        run_id="RUN-TRACE-01",
        task_id="T001",
        event_type=EventType.TOOL_CALL,
        tool_name="load_dataset",
        tool_input={"path": "data.csv"},
        tool_output={"rows": 100},
        decision="execute",
        decision_reason="Loaded rows",
    )

    events = trace_repo.get_trace_events("RUN-TRACE-01")
    assert len(events) == 2
    assert events[0].event_type == EventType.WORKFLOW_STARTED
    assert events[1].event_type == EventType.TOOL_CALL


def test_no_private_thoughts_stored(trace_repo):
    recorder = TraceRecorder(repository=trace_repo)
    recorder.record(
        run_id="RUN-TRACE-02",
        task_id="T002",
        event_type=EventType.TOOL_CALL,
        tool_name="eda_analysis",
        decision="execute",
        decision_reason="Standard analytical contract",
    )
    events = trace_repo.get_trace_events("RUN-TRACE-02")
    for ev in events:
        dump = ev.model_dump()
        assert "private_thought" not in dump
        assert "internal_reasoning_scratchpad" not in dump


def test_human_override_persistence(trace_repo):
    trace_repo.record_human_override(
        run_id="RUN-TRACE-03",
        task_id="T003",
        decision="approve",
        reason="Analyst verified clean anomaly",
        timestamp="2026-09-13T12:00:00Z",
    )
    overrides = trace_repo.get_human_overrides("RUN-TRACE-03")
    assert len(overrides) == 1
    assert overrides[0]["decision"] == "approve"
    assert overrides[0]["reason"] == "Analyst verified clean anomaly"
