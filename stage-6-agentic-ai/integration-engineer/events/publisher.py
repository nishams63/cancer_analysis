"""Helper for standardized event emission."""
import uuid
from typing import Optional, Dict, Any
from .event_types import IntegrationEvent, IntegrationEventType
from .event_bus import EventBus, get_global_event_bus


class EventPublisher:
    """Convenience publisher attached to a specific run session."""

    def __init__(self, run_id: str, bus: Optional[EventBus] = None):
        self.run_id = run_id
        self.bus = bus or get_global_event_bus()

    def emit(
        self,
        event_type: IntegrationEventType,
        message: str,
        task_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        severity: str = "INFO",
    ) -> IntegrationEvent:
        event = IntegrationEvent(
            event_id=f"EVT-{uuid.uuid4().hex[:8].upper()}",
            run_id=self.run_id,
            event_type=event_type,
            task_id=task_id,
            message=message,
            payload=payload or {},
            severity=severity,
        )
        self.bus.publish(event)
        return event
