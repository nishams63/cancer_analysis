"""Tests for SSE trace streaming generator."""
import pytest
import asyncio
from events.event_bus import EventBus
from events.event_types import IntegrationEventType, IntegrationEvent
from streaming.sse import sse_event_generator


@pytest.mark.anyio
async def test_sse_event_streaming():
    bus = EventBus()
    run_id = "RUN-STREAM-001"

    # Pre-populate one event
    evt1 = IntegrationEvent(
        event_id="EV-1",
        run_id=run_id,
        event_type=IntegrationEventType.RUN_STARTED,
        message="Started",
    )
    bus.publish(evt1)

    # Terminal event to end generator
    evt2 = IntegrationEvent(
        event_id="EV-2",
        run_id=run_id,
        event_type=IntegrationEventType.RUN_COMPLETED,
        message="Finished",
    )
    bus.publish(evt2)

    lines = []
    async for chunk in sse_event_generator(run_id=run_id, event_bus=bus):
        lines.append(chunk)
        if "RUN_COMPLETED" in chunk:
            break

    assert len(lines) >= 2
    assert any("RUN_STARTED" in l for l in lines)
    assert any("RUN_COMPLETED" in l for l in lines)
