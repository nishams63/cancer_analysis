from typing import Dict, Any, List, Optional
from ..utils.serialization import json_loads

class ScenarioStore:
    def __init__(self, result_store):
        self.result_store = result_store

    def get_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        with self.result_store.get_connection() as conn:
            row = conn.execute(
                """
                SELECT s.*, r.query_text, r.retrieval_status, r.concept_coverage,
                       n.narrative_text, n.validation_status as narrative_status,
                       e.realism_score, e.scenario_plausibility_score, e.rag_quality_score,
                       e.fidelity_score, e.faithfulness_score, e.difficulty_score, e.impact_score
                FROM scenarios s
                LEFT JOIN rag_results r ON s.scenario_id = r.scenario_id
                LEFT JOIN narratives n ON s.scenario_id = n.scenario_id
                LEFT JOIN evaluation_results e ON s.scenario_id = e.scenario_id
                WHERE s.scenario_id = ?
                """,
                (scenario_id,)
            ).fetchone()
            if not row:
                return None
            res = dict(row)
            if res.get("patient_json"):
                res["patient"] = json_loads(res["patient_json"])
            return res

    def list_scenarios(self, batch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.result_store.get_connection() as conn:
            if batch_id:
                rows = conn.execute(
                    """
                    SELECT s.scenario_id, s.batch_id, s.prompt_id, s.category, s.status, s.mutation_pattern,
                           e.realism_score, e.rag_quality_score, e.fidelity_score, e.faithfulness_score,
                           e.difficulty_score, e.impact_score
                    FROM scenarios s
                    LEFT JOIN evaluation_results e ON s.scenario_id = e.scenario_id
                    WHERE s.batch_id = ?
                    ORDER BY s.scenario_id ASC
                    """,
                    (batch_id,)
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT s.scenario_id, s.batch_id, s.prompt_id, s.category, s.status, s.mutation_pattern,
                           e.realism_score, e.rag_quality_score, e.fidelity_score, e.faithfulness_score,
                           e.difficulty_score, e.impact_score
                    FROM scenarios s
                    LEFT JOIN evaluation_results e ON s.scenario_id = e.scenario_id
                    ORDER BY s.created_at DESC
                    """
                ).fetchall()
            return [dict(r) for r in rows]
