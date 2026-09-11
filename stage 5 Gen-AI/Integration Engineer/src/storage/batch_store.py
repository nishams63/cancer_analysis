import sqlite3
from typing import Dict, Any, List, Optional
from ..utils.serialization import json_dumps, json_loads

class BatchStore:
    def __init__(self, result_store):
        self.result_store = result_store

    def create_batch(self, batch_id: str, seed: int, requested_count: int, config_versions: Dict[str, Any], system_manifest: Dict[str, Any]) -> None:
        with self.result_store.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO batches 
                (batch_id, seed, requested_count, actual_count, start_time, end_time, status, config_versions, system_manifest)
                VALUES (?, ?, ?, 0, datetime('now'), NULL, 'CREATED', ?, ?)
                """,
                (batch_id, seed, requested_count, json_dumps(config_versions), json_dumps(system_manifest))
            )
            conn.commit()

    def update_status(self, batch_id: str, status: str, actual_count: Optional[int] = None) -> None:
        with self.result_store.get_connection() as conn:
            if status in ("COMPLETED", "FAILED", "PARTIAL_FAILURE"):
                conn.execute(
                    """
                    UPDATE batches 
                    SET status = ?, actual_count = COALESCE(?, actual_count), end_time = datetime('now')
                    WHERE batch_id = ?
                    """,
                    (status, actual_count, batch_id)
                )
            else:
                conn.execute(
                    "UPDATE batches SET status = ?, actual_count = COALESCE(?, actual_count) WHERE batch_id = ?",
                    (status, actual_count, batch_id)
                )
            conn.commit()

    def get_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        with self.result_store.get_connection() as conn:
            row = conn.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
            return dict(row) if row else None

    def list_batches(self) -> List[Dict[str, Any]]:
        with self.result_store.get_connection() as conn:
            rows = conn.execute("SELECT * FROM batches ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]
