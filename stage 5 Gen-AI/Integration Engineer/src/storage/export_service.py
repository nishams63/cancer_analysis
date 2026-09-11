import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from ..utils.logging import get_logger
from ..utils.serialization import json_dumps

logger = get_logger("ExportService")

class ExportService:
    def __init__(self, result_store, base_dir: Optional[Path] = None):
        self.result_store = result_store
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent.parent

    def export_all(self, batch_id: str) -> Dict[str, str]:
        results_dir = self.base_dir / "results"
        batches_file = results_dir / "batches" / "batch_results.jsonl"
        scenarios_file = results_dir / "scenarios" / "scenario_results.jsonl"
        failures_file = results_dir / "failures" / "failure_results.jsonl"
        rankings_file = results_dir / "rankings" / "wildcard_candidates.csv"

        for f in [batches_file, scenarios_file, failures_file, rankings_file]:
            f.parent.mkdir(parents=True, exist_ok=True)

        # 1. Export Batch
        with self.result_store.get_connection() as conn:
            batch_row = conn.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
            if batch_row:
                with open(batches_file, "a", encoding="utf-8") as bf:
                    bf.write(json.dumps(dict(batch_row)) + "\n")

            # 2. Export Scenarios
            scen_rows = conn.execute(
                """
                SELECT s.scenario_id, s.batch_id, s.prompt_id, s.category, s.status, s.mutation_pattern,
                       r.query_text, r.retrieval_status, r.concept_coverage,
                       n.narrative_text, n.validation_status as narrative_status,
                       e.realism_score, e.scenario_plausibility_score, e.rag_quality_score,
                       e.fidelity_score, e.faithfulness_score, e.difficulty_score, e.impact_score
                FROM scenarios s
                LEFT JOIN rag_results r ON s.scenario_id = r.scenario_id
                LEFT JOIN narratives n ON s.scenario_id = n.scenario_id
                LEFT JOIN evaluation_results e ON s.scenario_id = e.scenario_id
                WHERE s.batch_id = ?
                """,
                (batch_id,)
            ).fetchall()
            
            with open(scenarios_file, "w", encoding="utf-8") as sf:
                for sr in scen_rows:
                    sf.write(json.dumps(dict(sr)) + "\n")

            # 3. Export Failures
            fail_rows = conn.execute("SELECT * FROM failures WHERE batch_id = ?", (batch_id,)).fetchall()
            with open(failures_file, "w", encoding="utf-8") as ff:
                for fr in fail_rows:
                    ff.write(json.dumps(dict(fr)) + "\n")

            # 4. Export Rankings
            rank_rows = conn.execute("SELECT * FROM wildcard_rankings WHERE batch_id = ? ORDER BY rank ASC", (batch_id,)).fetchall()
            rank_dicts = [dict(rr) for rr in rank_rows]
            if rank_dicts:
                df_ranks = pd.DataFrame(rank_dicts)
                df_ranks.to_csv(rankings_file, index=False)
            else:
                with open(rankings_file, "w", encoding="utf-8", newline="") as rf:
                    writer = csv.writer(rf)
                    writer.writerow(["id", "batch_id", "rank", "scenario_id", "wildcard_score", "is_candidate", "reason_explanation"])

        logger.info(f"Exported batch {batch_id} to JSONL and CSV.")
        return {
            "batches": str(batches_file),
            "scenarios": str(scenarios_file),
            "failures": str(failures_file),
            "rankings": str(rankings_file)
        }
