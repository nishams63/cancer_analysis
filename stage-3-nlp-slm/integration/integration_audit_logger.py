"""
Observability and Immutable Audit Logging Module for Stage 3 -> Stage 4 Integration.
Maintains an append-only, tamper-evident audit trail with SHA-256 payload digests,
enabling post-hoc regulatory verification of any downstream treatment recommendation.
"""

from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import sqlite3
import hashlib
import json
import uuid
from pathlib import Path
import logging

from contract_validation import Stage3OutputPayload, export_canonical_json
from confidence_gate import RoutingResult
from failure_mode_handlers import DegradedEnvelope

logger = logging.getLogger("stage3_stage4.audit_logger")

DEFAULT_AUDIT_DB_PATH = Path(__file__).resolve().parent / "integration_audit.db"
GENESIS_HASH = "0" * 64


from contextlib import contextmanager


@dataclass
class AuditEntry:
    audit_id: str
    document_id: str
    patient_id: str
    document_hash: str
    payload_hash: str
    urgency_predicted_class: str
    urgency_confidence: float
    hazard_predicted_class: str
    hazard_confidence: float
    entity_count: int
    routing_decision: Literal["STAGE_4_AUTOMATED", "HUMAN_REVIEW"]
    routing_reasons: List[str]
    model_version: str
    ner_model_version: str
    latency_ms_breakdown: Dict[str, float]
    status: str
    timestamp: str
    prev_hash: str
    entry_hash: str


class IntegrationAuditLogger:
    """
    Append-only tamper-evident audit logger for clinical NLP handoffs.
    Uses SQLite with WAL mode and cryptographic hash-chaining.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or DEFAULT_AUDIT_DB_PATH)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create audit log schema and indices if not present."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audit_id TEXT UNIQUE NOT NULL,
                    document_id TEXT NOT NULL,
                    patient_id TEXT NOT NULL,
                    document_hash TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    urgency_predicted_class TEXT NOT NULL,
                    urgency_confidence REAL NOT NULL,
                    hazard_predicted_class TEXT NOT NULL,
                    hazard_confidence REAL NOT NULL,
                    entity_count INTEGER NOT NULL,
                    routing_decision TEXT NOT NULL,
                    routing_reasons TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    ner_model_version TEXT NOT NULL,
                    latency_ms_breakdown TEXT NOT NULL,
                    status TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    entry_hash TEXT NOT NULL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_doc ON audit_log (document_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_patient ON audit_log (patient_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_routing ON audit_log (routing_decision);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log (timestamp);")
            conn.commit()

    def _get_last_entry_hash(self, conn: sqlite3.Connection) -> str:
        cur = conn.execute("SELECT entry_hash FROM audit_log ORDER BY row_id DESC LIMIT 1;")
        row = cur.fetchone()
        return row["entry_hash"] if row else GENESIS_HASH

    @staticmethod
    def compute_sha256(content: str) -> str:
        """Computes lowercase hex SHA-256 string."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def log_handoff(
        self,
        envelope: DegradedEnvelope,
        raw_text: str,
        latency_breakdown: Optional[Dict[str, float]] = None
    ) -> AuditEntry:
        """
        Records a completed handoff or degraded failure into the tamper-evident audit store.
        """
        audit_id = str(uuid.uuid4())
        doc_hash = self.compute_sha256(raw_text)
        now_ts = datetime.now(timezone.utc).isoformat()
        latency = latency_breakdown or {"total_ms": 0.0}

        if envelope.payload:
            payload_canonical = export_canonical_json(envelope.payload)
            payload_hash = self.compute_sha256(payload_canonical)
            urg_class = envelope.payload.triage_urgency.predicted_class
            urg_conf = envelope.payload.triage_urgency.confidence
            haz_class = envelope.payload.toxicity_hazard.predicted_class
            haz_conf = envelope.payload.toxicity_hazard.confidence
            entity_count = len(envelope.payload.clinical_entities)
            model_ver = envelope.payload.model_version
            ner_ver = envelope.payload.ner_model_version
        else:
            payload_hash = "0" * 64
            urg_class = "UNKNOWN"
            urg_conf = 0.0
            haz_class = "UNKNOWN"
            haz_conf = 0.0
            entity_count = 0
            model_ver = "NONE"
            ner_ver = "NONE"

        routing_reasons = (
            envelope.routing_result.routing_reasons
            if envelope.routing_result
            else [envelope.degradation_reason]
        )

        with self._get_connection() as conn:
            prev_hash = self._get_last_entry_hash(conn)

            # Compute entry hash for cryptographic hash-chaining
            block_content = (
                f"{prev_hash}|{audit_id}|{envelope.document_id}|{envelope.patient_id}|"
                f"{doc_hash}|{payload_hash}|{urg_class}|{urg_conf}|{haz_class}|{haz_conf}|"
                f"{envelope.destination}|{now_ts}"
            )
            entry_hash = self.compute_sha256(block_content)

            conn.execute("""
                INSERT INTO audit_log (
                    audit_id, document_id, patient_id, document_hash, payload_hash,
                    urgency_predicted_class, urgency_confidence,
                    hazard_predicted_class, hazard_confidence, entity_count,
                    routing_decision, routing_reasons, model_version, ner_model_version,
                    latency_ms_breakdown, status, timestamp, prev_hash, entry_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                audit_id,
                envelope.document_id,
                envelope.patient_id,
                doc_hash,
                payload_hash,
                urg_class,
                urg_conf,
                haz_class,
                haz_conf,
                entity_count,
                envelope.destination,
                json.dumps(routing_reasons),
                model_ver,
                ner_ver,
                json.dumps(latency),
                envelope.status,
                now_ts,
                prev_hash,
                entry_hash
            ))
            conn.commit()

        entry = AuditEntry(
            audit_id=audit_id,
            document_id=envelope.document_id,
            patient_id=envelope.patient_id,
            document_hash=doc_hash,
            payload_hash=payload_hash,
            urgency_predicted_class=urg_class,
            urgency_confidence=urg_conf,
            hazard_predicted_class=haz_class,
            hazard_confidence=haz_conf,
            entity_count=entity_count,
            routing_decision=envelope.destination,
            routing_reasons=routing_reasons,
            model_version=model_ver,
            ner_model_version=ner_ver,
            latency_ms_breakdown=latency,
            status=envelope.status,
            timestamp=now_ts,
            prev_hash=prev_hash,
            entry_hash=entry_hash
        )
        logger.info(f"Handoff logged [audit_id={audit_id}, doc={envelope.document_id}, dest={envelope.destination}]")
        return entry

    def query_by_document_id(self, document_id: str) -> List[Dict[str, Any]]:
        """Queries audit entries by document identifier."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM audit_log WHERE document_id = ? ORDER BY row_id ASC;", (document_id,))
            return [dict(r) for r in cur.fetchall()]

    def query_by_patient_id(self, patient_id: str) -> List[Dict[str, Any]]:
        """Queries audit entries by patient construct identifier."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM audit_log WHERE patient_id = ? ORDER BY row_id ASC;", (patient_id,))
            return [dict(r) for r in cur.fetchall()]

    def query_by_routing(self, destination: Literal["STAGE_4_AUTOMATED", "HUMAN_REVIEW"]) -> List[Dict[str, Any]]:
        """Queries audit entries by destination routing."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM audit_log WHERE routing_decision = ? ORDER BY row_id DESC;", (destination,))
            return [dict(r) for r in cur.fetchall()]

    def verify_audit_integrity(self) -> Dict[str, Any]:
        """
        Verifies the cryptographic tamper-evident hash chain across all log entries.
        Returns verification status and counts.
        """
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM audit_log ORDER BY row_id ASC;")
            rows = cur.fetchall()

        if not rows:
            return {"verified": True, "total_records": 0, "status": "EMPTY"}

        expected_prev = GENESIS_HASH
        for idx, row in enumerate(rows):
            if row["prev_hash"] != expected_prev:
                return {
                    "verified": False,
                    "total_records": len(rows),
                    "failed_row_id": row["row_id"],
                    "reason": f"Hash chain broken at row {row['row_id']}: expected prev_hash {expected_prev}, got {row['prev_hash']}"
                }

            block_content = (
                f"{row['prev_hash']}|{row['audit_id']}|{row['document_id']}|{row['patient_id']}|"
                f"{row['document_hash']}|{row['payload_hash']}|{row['urgency_predicted_class']}|"
                f"{row['urgency_confidence']}|{row['hazard_predicted_class']}|{row['hazard_confidence']}|"
                f"{row['routing_decision']}|{row['timestamp']}"
            )
            recomputed = self.compute_sha256(block_content)
            if recomputed != row["entry_hash"]:
                return {
                    "verified": False,
                    "total_records": len(rows),
                    "failed_row_id": row["row_id"],
                    "reason": f"Content tampering detected at row {row['row_id']}: entry_hash mismatch."
                }
            expected_prev = row["entry_hash"]

        return {
            "verified": True,
            "total_records": len(rows),
            "status": "ALL_RECORDS_VERIFIED_TAMPER_FREE",
            "head_hash": expected_prev
        }
