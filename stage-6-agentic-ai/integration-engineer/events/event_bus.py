"""Thread-safe In-Memory Event Bus for AADA integration."""
from __future__ import annotations
import asyncio
from typing import Dict, List, Callable, Any, Optional
from collections import defaultdict
import threading

from .event_types import IntegrationEvent, IntegrationEventType


class EventBus:
    """Asynchronous and synchronous in-memory publish/subscribe event bus."""

    def __init__(self):
        self._subscribers: Dict[Optional[str], List[Callable[[IntegrationEvent], Any]]] = defaultdict(list)
        self._history: Dict[str, List[IntegrationEvent]] = defaultdict(list)
        self._queues: Dict[str, List[asyncio.Queue]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable[[IntegrationEvent], Any], run_id: Optional[str] = None) -> None:
        """Register a callback for events. If run_id is None, receives all events."""
        with self._lock:
            self._subscribers[run_id].append(callback)

    def unsubscribe(self, callback: Callable[[IntegrationEvent], Any], run_id: Optional[str] = None) -> None:
        with self._lock:
            if callback in self._subscribers[run_id]:
                self._subscribers[run_id].remove(callback)

    def publish(self, event: IntegrationEvent) -> None:
        """Publish an event to subscribers and store in run history."""
        with self._lock:
            self._history[event.run_id].append(event)
            # Notify global subscribers
            global_subs = list(self._subscribers[None])
            # Notify run-specific subscribers
            run_subs = list(self._subscribers[event.run_id])
            target_queues = list(self._queues[event.run_id])

        for sub in global_subs + run_subs:
            try:
                sub(event)
            except Exception:
                pass

        # Push to async queues for streaming (SSE / WebSockets)
        for q in target_queues:
            try:
                q.put_nowait(event)
            except Exception:
                pass

    def get_events(self, run_id: str) -> List[IntegrationEvent]:
        """Retrieve full event log for a given run."""
        with self._lock:
            return list(self._history.get(run_id, []))

    def register_queue(self, run_id: str, queue: asyncio.Queue) -> None:
        """Register an asyncio queue to receive real-time events for SSE/WebSocket streaming."""
        with self._lock:
            self._queues[run_id].append(queue)

    def unregister_queue(self, run_id: str, queue: asyncio.Queue) -> None:
        with self._lock:
            if queue in self._queues[run_id]:
                self._queues[run_id].remove(queue)

    def clear(self) -> None:
        with self._lock:
            self._subscribers.clear()
            self._history.clear()
            self._queues.clear()


_GLOBAL_EVENT_BUS = EventBus()

def get_global_event_bus() -> EventBus:
    return _GLOBAL_EVENT_BUS
