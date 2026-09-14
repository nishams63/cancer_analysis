"""Server-Sent Events (SSE) generator for real-time trace streaming."""
import asyncio
import json
from typing import AsyncGenerator
from events.event_bus import EventBus, get_global_event_bus
from events.event_types import IntegrationEvent, IntegrationEventType


async def sse_event_generator(
    run_id: str,
    event_bus: EventBus = None,
    keepalive_interval: float = 15.0,
) -> AsyncGenerator[str, None]:
    """Yield formatted Server-Sent Events for a given run session."""
    bus = event_bus or get_global_event_bus()
    queue = asyncio.Queue()
    bus.register_queue(run_id, queue)

    try:
        # 1. Yield backlog of existing events
        backlog = bus.get_events(run_id)
        for evt in backlog:
            yield f"event: {evt.event_type.value}\ndata: {evt.model_dump_json()}\n\n"

        # 2. Stream live incoming events
        while True:
            try:
                evt: IntegrationEvent = await asyncio.wait_for(queue.get(), timeout=keepalive_interval)
                yield f"event: {evt.event_type.value}\ndata: {evt.model_dump_json()}\n\n"
                
                # Check for terminal lifecycle events
                if evt.event_type in (
                    IntegrationEventType.RUN_COMPLETED,
                    IntegrationEventType.RUN_FAILED,
                    IntegrationEventType.RUN_CANCELLED,
                ):
                    break
            except asyncio.TimeoutError:
                # Send keepalive ping to maintain connection
                yield ": keepalive\n\n"

    finally:
        bus.unregister_queue(run_id, queue)
