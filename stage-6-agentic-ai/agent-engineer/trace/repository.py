"""Trace Repository managing SQLite persistence of run traces."""
import sqlite3
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from schemas.agent import AgentState, RunStatus
from schemas.trace import TraceEvent, EventType
from trace.models import CREATE_TRACE_TABLES_SQL


class TraceRepository:
    """Thread-safe SQLite storage for runs, trace events, and human reviews."""

    def __init__(self, db_path: Optional[Path | str] = None):
        if db_path is None:
            self.db_path = Path(__file__).resolve().parent / "trace_history.db"
        else:
            self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(CREATE_TRACE_TABLES_SQL)
            conn.commit()

    def save_run(self, state: AgentState) -> None:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, workflow_id, status, started_at, completed_at, total_tasks, metrics_json, state_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.run_id,
                    state.workflow_id,
                    state.status.value,
                    state.metadata.get("started_at", ""),
                    state.metadata.get("completed_at", None),
                    state.metrics.total_tasks,
                    state.metrics.model_dump_json(),
                    state.model_dump_json(),
                ),
            )
            conn.commit()

    def save_event(self, event: TraceEvent) -> None:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO trace_events (
                    trace_id, run_id, timestamp, task_id, event_type, tool_name,
                    tool_input_json, tool_output_json, decision, decision_reason,
                    confidence, status, error_json, next_task_id, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.trace_id,
                    event.run_id,
                    event.timestamp,
                    event.task_id,
                    event.event_type.value,
                    event.tool_name,
                    json.dumps(event.tool_input) if event.tool_input else None,
                    json.dumps(event.tool_output_summary) if event.tool_output_summary else None,
                    event.decision,
                    event.decision_reason,
                    event.confidence,
                    event.status,
                    json.dumps(event.error) if event.error else None,
                    event.next_task_id,
                    json.dumps(event.metadata),
                ),
            )
            conn.commit()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = cur.fetchone()
            if not row:
                return None
            return {
                "run_id": row["run_id"],
                "workflow_id": row["workflow_id"],
                "status": row["status"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "total_tasks": row["total_tasks"],
                "metrics": json.loads(row["metrics_json"]),
                "state": json.loads(row["state_json"]),
            }

    def get_trace(self, run_id: str) -> List[TraceEvent]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM trace_events WHERE run_id = ? ORDER BY timestamp ASC", (run_id,))
            events = []
            for row in cur.fetchall():
                events.append(
                    TraceEvent(
                        trace_id=row["trace_id"],
                        run_id=row["run_id"],
                        timestamp=row["timestamp"],
                        task_id=row["task_id"],
                        event_type=EventType(row["event_type"]),
                        tool_name=row["tool_name"],
                        tool_input=json.loads(row["tool_input_json"]) if row["tool_input_json"] else None,
                        tool_output_summary=json.loads(row["tool_output_json"]) if row["tool_output_json"] else None,
                        decision=row["decision"],
                        decision_reason=row["decision_reason"],
                        confidence=row["confidence"],
                        status=row["status"],
                        error=json.loads(row["error_json"]) if row["error_json"] else None,
                        next_task_id=row["next_task_id"],
                        metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
                    )
                )
            return events

    get_trace_events = get_trace

    def get_human_overrides(self, run_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM human_overrides WHERE run_id = ? ORDER BY timestamp ASC", (run_id,))
            return [dict(row) for row in cur.fetchall()]

    def record_human_override(
        self, run_id: str, task_id: str, decision: str, reason: str, timestamp: str
    ) -> None:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO human_overrides (run_id, task_id, decision, reason, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, task_id, decision, reason, timestamp),
            )
            conn.commit()
