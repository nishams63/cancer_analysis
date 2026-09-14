"""Tests for the internal EventBus and event publisher."""
from events.event_types import IntegrationEventType, IntegrationEvent
from events.event_bus import EventBus
from events.publisher import EventPublisher


def test_event_bus_publish_and_subscribe():
    bus = EventBus()
    received = []

    def callback(evt: IntegrationEvent):
        received.append(evt)

    bus.subscribe(callback, run_id="RUN-001")

    pub = EventPublisher(run_id="RUN-001", bus=bus)
    pub.emit(
        event_type=IntegrationEventType.TASK_STARTED,
        message="Task T001 started",
        task_id="T001",
    )

    assert len(received) == 1
    assert received[0].event_type == IntegrationEventType.TASK_STARTED
    assert received[0].task_id == "T001"
    assert len(bus.get_events("RUN-001")) == 1


def test_event_bus_run_isolation():
    bus = EventBus()
    run1_events = []
    run2_events = []

    bus.subscribe(lambda e: run1_events.append(e), run_id="RUN-1")
    bus.subscribe(lambda e: run2_events.append(e), run_id="RUN-2")

    pub1 = EventPublisher(run_id="RUN-1", bus=bus)
    pub2 = EventPublisher(run_id="RUN-2", bus=bus)

    pub1.emit(IntegrationEventType.RUN_STARTED, "Started 1")
    pub2.emit(IntegrationEventType.RUN_STARTED, "Started 2")

    assert len(run1_events) == 1
    assert run1_events[0].run_id == "RUN-1"
    assert len(run2_events) == 1
    assert run2_events[0].run_id == "RUN-2"
