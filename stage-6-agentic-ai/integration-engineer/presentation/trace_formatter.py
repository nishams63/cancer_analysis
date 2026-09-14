"""Sanitizes and formats execution traces for human and API consumption."""
from typing import List, Dict, Any
from agent_engineer.schemas.trace import TraceEvent


class TraceFormatter:
    """Sanitizes raw TraceEvents to ensure no private reasoning or secrets are exposed."""

    SENSITIVE_KEYS = {"thought", "scratchpad", "chain_of_thought", "api_key", "secret", "password", "token"}

    @classmethod
    def sanitize_payload(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {
                k: cls.sanitize_payload(v)
                for k, v in data.items()
                if not any(sk in k.lower() for sk in cls.SENSITIVE_KEYS)
            }
        elif isinstance(data, list):
            return [cls.sanitize_payload(i) for i in data]
        return data

    @classmethod
    def format_event(cls, event: TraceEvent) -> Dict[str, Any]:
        """Convert a TraceEvent into a safe, client-facing timeline entry."""
        tool_name = event.tool_name or "workflow"
        message = event.decision_reason or f"Event {event.event_type.value} occurred."

        return {
            "trace_id": event.trace_id,
            "timestamp": event.timestamp,
            "task_id": event.task_id,
            "event_type": event.event_type.value,
            "tool_name": tool_name,
            "summary": message,
            "status": event.status,
            "input_summary": cls.sanitize_payload(event.tool_input),
            "output_summary": cls.sanitize_payload(event.tool_output_summary),
        }

    @classmethod
    def format_trace(cls, events: List[TraceEvent]) -> List[Dict[str, Any]]:
        return [cls.format_event(e) for e in events]
