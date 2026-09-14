"""Tests ensuring security invariants: no private thoughts or secrets exposed."""
from presentation.trace_formatter import TraceFormatter
from agent_engineer.schemas.trace import TraceEvent, EventType


def test_no_private_thoughts_or_secrets_exposed():
    sensitive_event = TraceEvent(
        trace_id="TR-SECRET",
        run_id="R-1",
        timestamp="2026-09-13T12:00:00Z",
        event_type=EventType.TOOL_CALL,
        tool_name="eda_analysis",
        tool_input={"api_key": "SECRET-NVIDIA-123", "param": "value"},
        tool_output_summary={"scratchpad": "private internal reasoning", "result": 42},
        decision="done",
        decision_reason="Analyzed dataset",
    )

    formatted = TraceFormatter.format_event(sensitive_event)

    # Verify input_summary sanitized
    assert "api_key" not in formatted["input_summary"]
    assert formatted["input_summary"]["param"] == "value"

    # Verify output_summary sanitized
    assert "scratchpad" not in formatted["output_summary"]
    assert formatted["output_summary"]["result"] == 42
