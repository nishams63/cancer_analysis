from typing import Dict, Any, List, Optional

class FailureStore:
    def __init__(self, result_store):
        self.result_store = result_store

    def list_failures(self, batch_id: Optional[str] = None, failure_code: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.result_store.get_connection() as conn:
            query = "SELECT * FROM failures WHERE 1=1"
            params = []
            if batch_id:
                query += " AND batch_id = ?"
                params.append(batch_id)
            if failure_code:
                query += " AND failure_code = ?"
                params.append(failure_code)
            query += " ORDER BY id ASC"
            rows = conn.execute(query, tuple(params)).fetchall()
            return [dict(r) for r in rows]

    def save_wildcard_ranking(self, batch_id: str, rank: int, scenario_id: str, wildcard_score: float, is_candidate: bool, reason_explanation: str) -> None:
        with self.result_store.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO wildcard_rankings (batch_id, rank, scenario_id, wildcard_score, is_candidate, reason_explanation)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (batch_id, rank, scenario_id, wildcard_score, 1 if is_candidate else 0, reason_explanation)
            )
            conn.commit()

    def list_rankings(self, batch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.result_store.get_connection() as conn:
            if batch_id:
                rows = conn.execute(
                    "SELECT * FROM wildcard_rankings WHERE batch_id = ? ORDER BY rank ASC", (batch_id,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM wildcard_rankings ORDER BY id DESC").fetchall()
            return [dict(r) for r in rows]
