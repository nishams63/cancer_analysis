"""In-memory and persistent trace event recorder."""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from schemas.trace import TraceEvent, EventType
from trace.repository import TraceRepository


class TraceRecorder:
    """Captures and persists auditable execution events."""

    def __init__(self, repository: Optional[TraceRepository] = None):
        self.repo = repository or TraceRepository()
        self.events: List[TraceEvent] = []

    def record(
        self,
        run_id: str,
        event_type: EventType,
        task_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_input: Optional[Dict[str, Any]] = None,
        tool_output: Optional[Dict[str, Any]] = None,
        decision: Optional[str] = None,
        decision_reason: Optional[str] = None,
        confidence: Optional[float] = None,
        status: str = "success",
        error: Optional[Dict[str, Any]] = None,
        next_task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceEvent:
        now_iso = datetime.utcnow().isoformat() + "Z"
        trace_id = f"TRACE-{uuid.uuid4().hex[:8].upper()}"

        event = TraceEvent(
            trace_id=trace_id,
            run_id=run_id,
            timestamp=now_iso,
            task_id=task_id,
            event_type=event_type,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output_summary=tool_output,
            decision=decision,
            decision_reason=decision_reason,
            confidence=confidence,
            status=status,
            error=error,
            next_task_id=next_task_id,
            metadata=metadata or {},
        )

        self.events.append(event)
        self.repo.save_event(event)
        return event
