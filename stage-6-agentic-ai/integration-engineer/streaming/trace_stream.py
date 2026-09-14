"""Bridges Agent Engineer trace records into the Integration EventBus."""
from typing import Optional, List
from events.event_bus import EventBus, get_global_event_bus
from events.publisher import EventPublisher
from events.event_types import IntegrationEventType
from agent_engineer.schemas.trace import TraceEvent, EventType


class TraceStreamBridge:
    """Converts low-level agent trace events into integration events."""

    MAPPING = {
        EventType.TASK_STARTED: IntegrationEventType.TASK_STARTED,
        EventType.TASK_COMPLETED: IntegrationEventType.TASK_COMPLETED,
        EventType.TASK_FAILED: IntegrationEventType.TASK_FAILED,
        EventType.TASK_SKIPPED: IntegrationEventType.TASK_SKIPPED,
        EventType.TOOL_CALL: IntegrationEventType.TOOL_COMPLETED,
        EventType.OBSERVATION: IntegrationEventType.KNOWLEDGE_RETRIEVED,
        EventType.BRANCH_DECISION: IntegrationEventType.BRANCH_EVALUATED,
        EventType.ESCALATION: IntegrationEventType.ESCALATION_TRIGGERED,
        EventType.HUMAN_OVERRIDE: IntegrationEventType.HUMAN_OVERRIDE,
    }

    def __init__(self, run_id: str, bus: Optional[EventBus] = None):
        self.run_id = run_id
        self.publisher = EventPublisher(run_id=run_id, bus=bus or get_global_event_bus())

    def bridge_event(self, trace_event: TraceEvent) -> None:
        int_type = self.MAPPING.get(trace_event.event_type)
        if not int_type:
            return

        msg = trace_event.decision_reason or f"Executed {trace_event.event_type.value}"
        self.publisher.emit(
            event_type=int_type,
            message=msg,
            task_id=trace_event.task_id,
            payload={
                "tool_name": trace_event.tool_name,
                "status": trace_event.status,
            },
            severity="WARNING" if trace_event.status == "warning" else "INFO",
        )
