import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from ..utils.logging import get_logger
from ..utils.serialization import json_dumps

logger = get_logger("ResultStore")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    seed INTEGER,
    requested_count INTEGER,
    actual_count INTEGER,
    start_time TEXT,
    end_time TEXT,
    status TEXT,
    config_versions TEXT,
    system_manifest TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scenarios (
    scenario_id TEXT PRIMARY KEY,
    batch_id TEXT,
    prompt_id TEXT,
    category TEXT,
    status TEXT,
    mutation_pattern TEXT,
    patient_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(batch_id) REFERENCES batches(batch_id)
);

CREATE TABLE IF NOT EXISTS rag_results (
    scenario_id TEXT PRIMARY KEY,
    batch_id TEXT,
    query_text TEXT,
    retrieved_chunks_json TEXT,
    retrieval_status TEXT,
    concept_coverage REAL,
    FOREIGN KEY(scenario_id) REFERENCES scenarios(scenario_id)
);

CREATE TABLE IF NOT EXISTS narratives (
    scenario_id TEXT PRIMARY KEY,
    batch_id TEXT,
    narrative_text TEXT,
    validation_status TEXT,
    faithfulness_score REAL,
    FOREIGN KEY(scenario_id) REFERENCES scenarios(scenario_id)
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    scenario_id TEXT PRIMARY KEY,
    batch_id TEXT,
    realism_score REAL,
    scenario_plausibility_score REAL,
    rag_quality_score REAL,
    fidelity_score REAL,
    faithfulness_score REAL,
    difficulty_score REAL,
    impact_score REAL,
    details_json TEXT,
    FOREIGN KEY(scenario_id) REFERENCES scenarios(scenario_id)
);

CREATE TABLE IF NOT EXISTS stage_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario_id TEXT,
    batch_id TEXT,
    stage_name TEXT,
    status TEXT,
    prediction_json TEXT,
    confidence_json TEXT,
    latency_ms REAL,
    error_message TEXT,
    FOREIGN KEY(scenario_id) REFERENCES scenarios(scenario_id)
);

CREATE TABLE IF NOT EXISTS failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario_id TEXT,
    batch_id TEXT,
    stage_name TEXT,
    failure_code TEXT,
    confidence REAL,
    evidence_text TEXT,
    FOREIGN KEY(scenario_id) REFERENCES scenarios(scenario_id)
);

CREATE TABLE IF NOT EXISTS counterfactual_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario_id TEXT,
    batch_id TEXT,
    factual_pred TEXT,
    counterfactual_pred TEXT,
    sensitivity_detected INTEGER,
    instability_detected INTEGER,
    details_json TEXT,
    FOREIGN KEY(scenario_id) REFERENCES scenarios(scenario_id)
);

CREATE TABLE IF NOT EXISTS wildcard_rankings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT,
    rank INTEGER,
    scenario_id TEXT,
    wildcard_score REAL,
    is_candidate INTEGER,
    reason_explanation TEXT,
    FOREIGN KEY(scenario_id) REFERENCES scenarios(scenario_id)
);
"""

class ResultStore:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base = Path(__file__).resolve().parent.parent.parent
            self.db_path = base / "results" / "stage5.db"
        else:
            self.db_path = Path(db_path)
            
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self):
        with self.get_connection() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    def save_consolidated_record(self, record: Dict[str, Any]) -> None:
        scenario_id = record.get("scenario_id")
        batch_id = record.get("batch_id")
        scenario = record.get("scenario", {})
        rag = record.get("rag", {})
        narrative = record.get("narrative", {})
        eval_res = record.get("evaluation", {})
        stages = record.get("stages", {})
        failures = record.get("failures", [])
        cf = record.get("counterfactuals", [])

        with self.get_connection() as conn:
            try:
                # 1. Scenarios
                muts = scenario.get("mutations", [])
                mut_str = ", ".join(muts) if isinstance(muts, list) else str(muts)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO scenarios 
                    (scenario_id, batch_id, prompt_id, category, status, mutation_pattern, patient_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        scenario_id,
                        batch_id,
                        scenario.get("scenario_id", ""),
                        scenario.get("clinical_archetype", "Standard"),
                        record.get("status", "SUCCESS"),
                        mut_str,
                        json_dumps(scenario)
                    )
                )

                # 2. RAG
                conn.execute(
                    """
                    INSERT OR REPLACE INTO rag_results
                    (scenario_id, batch_id, query_text, retrieved_chunks_json, retrieval_status, concept_coverage)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        scenario_id,
                        batch_id,
                        rag.get("query", ""),
                        json_dumps(rag.get("retrieved_chunks", [])),
                        rag.get("status", "SUCCESS"),
                        float(rag.get("concept_coverage", 1.0))
                    )
                )

                # 3. Narrative
                conn.execute(
                    """
                    INSERT OR REPLACE INTO narratives
                    (scenario_id, batch_id, narrative_text, validation_status, faithfulness_score)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        scenario_id,
                        batch_id,
                        narrative.get("narrative_text", ""),
                        narrative.get("validation_status", "VALID"),
                        float(eval_res.get("narrative_faithfulness", {}).get("faithfulness_score", 0.90))
                    )
                )

                # 4. Evaluation Results
                conn.execute(
                    """
                    INSERT OR REPLACE INTO evaluation_results
                    (scenario_id, batch_id, realism_score, scenario_plausibility_score, rag_quality_score,
                     fidelity_score, faithfulness_score, difficulty_score, impact_score, details_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        scenario_id,
                        batch_id,
                        float(eval_res.get("realism", {}).get("score", 0.85)),
                        float(eval_res.get("scenario_plausibility", {}).get("score", 0.85)),
                        float(eval_res.get("rag_quality", {}).get("score", 0.90)),
                        float(eval_res.get("fidelity", {}).get("fidelity_score", 1.0)),
                        float(eval_res.get("narrative_faithfulness", {}).get("faithfulness_score", 0.90)),
                        float(eval_res.get("difficulty", {}).get("difficulty_score", 50.0)),
                        float(eval_res.get("impact", {}).get("impact_score", 50.0)),
                        json_dumps(eval_res)
                    )
                )

                # 5. Stage Results
                for stage_name, sdata in stages.items():
                    conn.execute(
                        """
                        INSERT INTO stage_results
                        (scenario_id, batch_id, stage_name, status, prediction_json, confidence_json, latency_ms, error_message)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            scenario_id,
                            batch_id,
                            stage_name,
                            sdata.get("status", "SUCCESS"),
                            json_dumps(sdata.get("prediction", {})),
                            json_dumps(sdata.get("confidence", {})),
                            float(sdata.get("latency_ms", 0.0)),
                            sdata.get("error")
                        )
                    )

                # 6. Failures
                for f in failures:
                    conn.execute(
                        """
                        INSERT INTO failures
                        (scenario_id, batch_id, stage_name, failure_code, confidence, evidence_text)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            scenario_id,
                            batch_id,
                            f.get("stage", "stage1"),
                            f.get("code", "F01"),
                            float(f.get("confidence", 0.80)),
                            f.get("evidence", "")
                        )
                    )

                # 7. Counterfactuals
                for c in cf:
                    conn.execute(
                        """
                        INSERT INTO counterfactual_results
                        (scenario_id, batch_id, factual_pred, counterfactual_pred, sensitivity_detected, instability_detected, details_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            scenario_id,
                            batch_id,
                            str(c.get("factual_pred", "")),
                            str(c.get("counterfactual_pred", "")),
                            1 if c.get("sensitivity_detected", False) else 0,
                            1 if c.get("instability_detected", False) else 0,
                            json_dumps(c)
                        )
                    )

                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to persist consolidated record {scenario_id}: {e}")
                raise e
