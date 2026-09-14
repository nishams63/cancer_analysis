"""WebSocket manager for real-time trace broadcast."""
from typing import Dict, List
import json
from fastapi import WebSocket
from events.event_types import IntegrationEvent


class WebSocketManager:
    """Manages active WebSocket connections subscribed to runs."""

    def __init__(self):
        self._active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, run_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        if run_id not in self._active_connections:
            self._active_connections[run_id] = []
        self._active_connections[run_id].append(websocket)

    def disconnect(self, run_id: str, websocket: WebSocket) -> None:
        if run_id in self._active_connections and websocket in self._active_connections[run_id]:
            self._active_connections[run_id].remove(websocket)

    async def broadcast_event(self, run_id: str, event: IntegrationEvent) -> None:
        conns = list(self._active_connections.get(run_id, []))
        data = event.model_dump_json()
        for ws in conns:
            try:
                await ws.send_text(data)
            except Exception:
                self.disconnect(run_id, ws)


_GLOBAL_WS_MANAGER = WebSocketManager()

def get_websocket_manager() -> WebSocketManager:
    return _GLOBAL_WS_MANAGER
