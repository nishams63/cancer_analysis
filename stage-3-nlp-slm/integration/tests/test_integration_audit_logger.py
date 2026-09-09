"""
Unit tests for Immutable Audit Logger (integration_audit_logger.py).
Tests append-only logging, query APIs, and cryptographic tamper detection.
"""

import pytest
import tempfile
import sqlite3
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integration"))

from contract_validation import (
    Stage3OutputPayload,
    TriageUrgencyPayload,
    ToxicityHazardPayload,
    ClinicalEntityPayload,
)
from failure_mode_handlers import DegradedEnvelope
from confidence_gate import RoutingResult
from integration_audit_logger import IntegrationAuditLogger, GENESIS_HASH


@pytest.fixture
def temp_audit_logger(tmp_path):
    db_file = tmp_path / "test_audit.db"
    return IntegrationAuditLogger(db_path=db_file)


def make_mock_envelope(doc_id="DOC-AUDIT-1", patient_id="PT-AUDIT-1", dest="STAGE_4_AUTOMATED"):
    payload = Stage3OutputPayload(
        schema_version="1.0.0",
        document_id=doc_id,
        patient_id=patient_id,
        inference_timestamp=datetime.now(timezone.utc).isoformat(),
        model_version="stage3-minilm-hybrid-v3.3",
        ner_model_version="stage3-ner-v1.0",
        triage_urgency=TriageUrgencyPayload(
            predicted_class="HIGH",
            confidence=0.91,
            class_probabilities={"HIGH": 0.91}
        ),
        toxicity_hazard=ToxicityHazardPayload(
            predicted_class="HEPATIC",
            confidence=0.87,
            class_probabilities={"HEPATIC": 0.87}
        ),
        clinical_entities=[
            ClinicalEntityPayload(
                start=0,
                end=11,
                label="DRUG_NAME",
                text="capecitabine",
                polarity="AFFIRMED"
            )
        ]
    )
    routing = RoutingResult(
        destination=dest,
        is_automated=(dest == "STAGE_4_AUTOMATED"),
        routing_reasons=["High confidence match"],
        effective_urgency_class="HIGH",
        effective_hazard_class="HEPATIC"
    )
    return DegradedEnvelope(
        status="SUCCESS" if dest == "STAGE_4_AUTOMATED" else "LOW_CONFIDENCE",
        destination=dest,
        document_id=doc_id,
        patient_id=patient_id,
        degradation_reason="Normal execution",
        payload=payload,
        routing_result=routing
    )


def test_audit_empty_integrity(temp_audit_logger):
    ver = temp_audit_logger.verify_audit_integrity()
    assert ver["verified"] is True
    assert ver["total_records"] == 0


def test_log_handoff_and_query(temp_audit_logger):
    env1 = make_mock_envelope(doc_id="DOC-101", patient_id="PT-001", dest="STAGE_4_AUTOMATED")
    raw_text = "Patient receiving capecitabine with elevated transaminases."
    latency = {"inference_ms": 18.2, "validation_ms": 1.1, "total_ms": 19.3}

    entry1 = temp_audit_logger.log_handoff(env1, raw_text, latency)
    assert entry1.document_id == "DOC-101"
    assert entry1.patient_id == "PT-001"
    assert entry1.routing_decision == "STAGE_4_AUTOMATED"
    assert entry1.prev_hash == GENESIS_HASH

    # Query by doc
    docs = temp_audit_logger.query_by_document_id("DOC-101")
    assert len(docs) == 1
    assert docs[0]["document_id"] == "DOC-101"

    # Query by patient
    pts = temp_audit_logger.query_by_patient_id("PT-001")
    assert len(pts) == 1
    assert pts[0]["patient_id"] == "PT-001"

    # Query by routing
    auto = temp_audit_logger.query_by_routing("STAGE_4_AUTOMATED")
    assert len(auto) == 1
    human = temp_audit_logger.query_by_routing("HUMAN_REVIEW")
    assert len(human) == 0


def test_hash_chain_tamper_detection(temp_audit_logger):
    # Log 3 entries
    for i in range(3):
        env = make_mock_envelope(doc_id=f"DOC-{i}", patient_id=f"PT-{i}")
        temp_audit_logger.log_handoff(env, f"Sample text note {i}")

    # Verify chain is clean
    ver = temp_audit_logger.verify_audit_integrity()
    assert ver["verified"] is True
    assert ver["total_records"] == 3

    # Tamper with row 2 in the database
    with temp_audit_logger._get_connection() as conn:
        conn.execute("UPDATE audit_log SET urgency_predicted_class = 'CRITICAL' WHERE row_id = 2;")
        conn.commit()

    # Verify detection catches tampering
    tamper_ver = temp_audit_logger.verify_audit_integrity()
    assert tamper_ver["verified"] is False
    assert "Content tampering detected" in tamper_ver["reason"]


def test_log_envelope_without_payload(temp_audit_logger):
    # e.g., timeout or malformed input where payload is None
    env = DegradedEnvelope(
        status="TIMEOUT",
        destination="HUMAN_REVIEW",
        document_id="DOC-TIMEOUT-001",
        patient_id="PT-TIMEOUT-001",
        degradation_reason="Inference timed out after 100ms",
        error_details="Stage3TimeoutError: exceeded deadline"
    )
    entry = temp_audit_logger.log_handoff(env, "Unprocessed text note")
    assert entry.status == "TIMEOUT"
    assert entry.routing_decision == "HUMAN_REVIEW"
    assert entry.payload_hash == "0" * 64
    assert entry.urgency_predicted_class == "UNKNOWN"

    # Query human review
    human_logs = temp_audit_logger.query_by_routing("HUMAN_REVIEW")
    assert len(human_logs) == 1
    assert human_logs[0]["document_id"] == "DOC-TIMEOUT-001"


def test_prev_hash_chain_break_detection(temp_audit_logger):
    # Log 2 entries
    for i in range(2):
        env = make_mock_envelope(doc_id=f"DOC-CHAIN-{i}", patient_id=f"PT-CHAIN-{i}")
        temp_audit_logger.log_handoff(env, f"Sample text {i}")

    # Break prev_hash of row 2 directly
    with temp_audit_logger._get_connection() as conn:
        conn.execute("UPDATE audit_log SET prev_hash = 'corrupted_prev_hash' WHERE row_id = 2;")
        conn.commit()

    ver = temp_audit_logger.verify_audit_integrity()
    assert ver["verified"] is False
    assert "Hash chain broken at row 2" in ver["reason"]

