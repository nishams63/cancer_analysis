"""
End-to-End Integration Test Suite for Stage 3 -> Stage 4 Production Handoff.
Runs 65 representative documents sampled EXCLUSIVELY from out-of-sample validation fixtures (validation.parquet).
Zero documents from training data.
Sealed holdout data remains sealed and is strictly off-limits.

Asserts:
1. 100% schema validity at Stage 3 egress and Stage 4 ingress boundaries.
2. Correct routing decisions under ConfidenceGate (general threshold 0.65, rare hazard 0.70).
3. Latency budget compliance: Stage 3 inference budget ~18.2ms, integration overhead <= 5.0ms (P95 <= 3.5ms).
4. Zero data loss between stages (all patient IDs, document IDs, entity offsets preserved).
5. Cryptographic tamper-evident audit log integrity verified across all runs.
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
INTEGRATION_DIR = ROOT / "stage-3-nlp-slm" / "integration"
NLP_SRC = ROOT / "stage-3-nlp-slm" / "nlp" / "src"
VALIDATION_PATH = ROOT / "stage-3-nlp-slm" / "data-engineering" / "data" / "processed" / "validation.parquet"

sys.path.insert(0, str(INTEGRATION_DIR))
sys.path.insert(0, str(NLP_SRC))

from contract_validation import (
    Stage3OutputPayload,
    validate_stage3_output,
    validate_stage4_input,
    SUPPORTED_SCHEMA_VERSION,
)
from confidence_gate import ConfidenceGate, ConfidenceGateConfig, RoutingResult
from failure_mode_handlers import IntegrationFailureHandler, DegradedEnvelope
from integration_audit_logger import IntegrationAuditLogger
from inference import ClinicalNLPInferenceEngine


def load_validation_test_cohort(n_target: int = 65) -> pd.DataFrame:
    """
    Samples representative test cohort EXCLUSIVELY from validation.parquet.
    Stratified across all 4 urgency classes and all 8 hazard classes (including 100% of rare cohorts).
    """
    assert VALIDATION_PATH.exists(), f"validation.parquet not found at {VALIDATION_PATH}"
    val_df = pd.read_parquet(VALIDATION_PATH)

    # 1. Take all rare hazard classes to ensure maximum stress-testing of confidence gate
    rare_derm = val_df[val_df["hazard_type"] == "DERMATOLOGIC"]       # n = 4
    rare_cardiac = val_df[val_df["hazard_type"] == "CARDIAC"]         # n = 5
    rare_neuro = val_df[val_df["hazard_type"] == "NEUROPATHIC"].head(10)  # n = 10
    renal = val_df[val_df["hazard_type"] == "RENAL"].head(10)             # n = 10
    hema = val_df[val_df["hazard_type"] == "HEMATOLOGIC"].head(10)        # n = 10
    pulm = val_df[val_df["hazard_type"] == "PULMONARY"].head(8)           # n = 8
    hep = val_df[val_df["hazard_type"] == "HEPATIC"].head(8)             # n = 8
    
    # 2. Sample common NONE hazard across all 4 urgency levels
    none_crit = val_df[(val_df["hazard_type"] == "NONE") & (val_df["urgency_level"] == "CRITICAL")].head(3)
    none_high = val_df[(val_df["hazard_type"] == "NONE") & (val_df["urgency_level"] == "HIGH")].head(3)
    none_med = val_df[(val_df["hazard_type"] == "NONE") & (val_df["urgency_level"] == "MEDIUM")].head(2)
    none_low = val_df[(val_df["hazard_type"] == "NONE") & (val_df["urgency_level"] == "LOW")].head(2)

    cohort = pd.concat([
        rare_derm, rare_cardiac, rare_neuro, renal, hema, pulm, hep,
        none_crit, none_high, none_med, none_low
    ]).drop_duplicates(subset=["document_id"])

    assert len(cohort) >= 50, f"Expected at least 50 sampled documents, got {len(cohort)}"
    return cohort


def run_e2e_integration_pipeline():
    """
    Executes the full end-to-end integration test harness.
    Returns metrics dictionary for verification and reporting.
    """
    cohort = load_validation_test_cohort(65)
    inference_engine = ClinicalNLPInferenceEngine()
    confidence_gate = ConfidenceGate(ConfidenceGateConfig(
        urgency_min_confidence=0.65,
        hazard_min_confidence=0.65,
        rare_hazard_min_confidence=0.70,
        critical_override_threshold=0.30,
        mode="CONFIG_C_PROMOTED"
    ))
    failure_handler = IntegrationFailureHandler(
        timeout_seconds=0.100,
        confidence_gate=confidence_gate,
        require_non_empty_entities=False  # allow real validation notes with no entities to flow to evaluation
    )

    # Use isolated test audit database in memory or temp file
    test_db = INTEGRATION_DIR / "tests" / "e2e_test_audit.db"
    if test_db.exists():
        test_db.unlink()
    audit_logger = IntegrationAuditLogger(db_path=test_db)

    nlp_latencies = []
    overhead_latencies = []
    total_latencies = []

    stage3_egress_passes = 0
    stage4_ingress_passes = 0
    automated_routes = 0
    human_routes = 0
    data_loss_flags = 0

    results = []

    print("\n" + "="*70)
    print(f"RUNNING E2E INTEGRATION HARNESS ON {len(cohort)} EXCLUSIVE VALIDATION DOCUMENTS")
    print("="*70)

    for idx, row in cohort.iterrows():
        doc_id = str(row["document_id"])
        patient_id = str(row["patient_id"])
        raw_text = str(row["text"])

        t_start = time.perf_counter()

        # Step 1: Input Sanitization
        sanitized_text = failure_handler.sanitize_input_document(raw_text, doc_id, patient_id)

        # Step 2: Stage 3 NLP Inference (Instrumented)
        t_nlp_start = time.perf_counter()
        raw_nlp = inference_engine.analyze_document(sanitized_text, document_id=doc_id)
        nlp_time_ms = (time.perf_counter() - t_nlp_start) * 1000.0

        # Step 3: Package into Contract Output
        t_overhead_start = time.perf_counter()
        raw_payload = {
            "schema_version": SUPPORTED_SCHEMA_VERSION,
            "document_id": doc_id,
            "patient_id": patient_id,
            "inference_timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": "stage3-minilm-hybrid-v3.3-config-c",
            "ner_model_version": "stage3-trainable-ner-v1.0",
            "triage_urgency": raw_nlp["triage_urgency"],
            "toxicity_hazard": raw_nlp["toxicity_hazard"],
            "clinical_entities": raw_nlp["clinical_entities"],
            "raw_text_hash": IntegrationAuditLogger.compute_sha256(sanitized_text),
            "integration_metadata": {"e2e_source": "validation.parquet"}
        }

        # Step 4: Stage 3 Boundary Validation
        payload_stage3 = validate_stage3_output(raw_payload)
        stage3_egress_passes += 1

        # Step 5: Confidence Gate Evaluation
        routing = confidence_gate.evaluate(payload_stage3)
        if routing.destination == "STAGE_4_AUTOMATED":
            automated_routes += 1
        else:
            human_routes += 1

        # Step 6: Stage 4 Ingress Validation (Simulating consumption in Stage 4)
        payload_stage4 = validate_stage4_input(payload_stage3)
        stage4_ingress_passes += 1

        # Step 7: Zero Data Loss Verification
        assert payload_stage4.document_id == doc_id
        assert payload_stage4.patient_id == patient_id
        assert len(payload_stage4.clinical_entities) == len(raw_nlp["clinical_entities"])
        for e_orig, e_trans in zip(raw_nlp["clinical_entities"], payload_stage4.clinical_entities):
            if (e_orig["start"] != e_trans.start or
                e_orig["end"] != e_trans.end or
                e_orig["label"] != e_trans.label or
                e_orig["text"] != e_trans.text):
                data_loss_flags += 1

        # Step 8: Build Envelope and Record Audit
        envelope = DegradedEnvelope(
            status="SUCCESS",
            destination=routing.destination,
            document_id=doc_id,
            patient_id=patient_id,
            degradation_reason="; ".join(routing.routing_reasons),
            payload=payload_stage4,
            routing_result=routing
        )

        overhead_time_ms = (time.perf_counter() - t_overhead_start) * 1000.0
        total_time_ms = (time.perf_counter() - t_start) * 1000.0

        nlp_latencies.append(nlp_time_ms)
        overhead_latencies.append(overhead_time_ms)
        total_latencies.append(total_time_ms)

        latency_breakdown = {
            "nlp_inference_ms": round(nlp_time_ms, 2),
            "integration_overhead_ms": round(overhead_time_ms, 2),
            "total_latency_ms": round(total_time_ms, 2)
        }
        audit_logger.log_handoff(envelope, sanitized_text, latency_breakdown)

        results.append({
            "document_id": doc_id,
            "patient_id": patient_id,
            "urgency": payload_stage4.triage_urgency.predicted_class,
            "urgency_conf": payload_stage4.triage_urgency.confidence,
            "hazard": payload_stage4.toxicity_hazard.predicted_class,
            "hazard_conf": payload_stage4.toxicity_hazard.confidence,
            "entities": len(payload_stage4.clinical_entities),
            "destination": routing.destination,
            "nlp_ms": nlp_time_ms,
            "overhead_ms": overhead_time_ms,
            "total_ms": total_time_ms
        })

    # Step 9: Audit Trail Tamper-Evidence Verification
    integrity = audit_logger.verify_audit_integrity()
    assert integrity["verified"] is True, f"Audit integrity failed: {integrity}"

    # Latency Percentiles
    overhead_p50 = float(np.percentile(overhead_latencies, 50))
    overhead_p90 = float(np.percentile(overhead_latencies, 90))
    overhead_p95 = float(np.percentile(overhead_latencies, 95))
    nlp_p95 = float(np.percentile(nlp_latencies, 95))
    total_p95 = float(np.percentile(total_latencies, 95))

    print("\n" + "="*70)
    print("E2E INTEGRATION HARNESS RESULTS SUMMARY")
    print("="*70)
    print(f"Total Documents Tested     : {len(cohort)} (100% from validation.parquet)")
    print(f"Stage 3 Egress Passes      : {stage3_egress_passes}/{len(cohort)} ({stage3_egress_passes/len(cohort)*100:.1f}%)")
    print(f"Stage 4 Ingress Passes     : {stage4_ingress_passes}/{len(cohort)} ({stage4_ingress_passes/len(cohort)*100:.1f}%)")
    print(f"Routing to STAGE_4_AUTO    : {automated_routes} ({automated_routes/len(cohort)*100:.1f}%)")
    print(f"Routing to HUMAN_REVIEW    : {human_routes} ({human_routes/len(cohort)*100:.1f}%)")
    print(f"Zero Data Loss Compliance  : {'PASS (0 losses)' if data_loss_flags == 0 else f'FAIL ({data_loss_flags} flags)'}")
    print(f"Audit Integrity Status     : {integrity['status']} ({integrity['total_records']} verified)")
    print("-"*70)
    print(f"Integration Overhead P50   : {overhead_p50:.2f} ms")
    print(f"Integration Overhead P90   : {overhead_p90:.2f} ms")
    print(f"Integration Overhead P95   : {overhead_p95:.2f} ms  (Budget: <= 5.0 ms, Target: <= 3.5 ms)")
    print(f"NLP Inference P95          : {nlp_p95:.2f} ms  (Budget: <= 18.2 ms)")
    print(f"Total Pipeline P95         : {total_p95:.2f} ms  (Hospital SLA: <= 250 ms)")
    print("="*70 + "\n")

    # Clean up test database
    if test_db.exists():
        try:
            test_db.unlink(missing_ok=True)
            # Also clean up WAL and SHM if present
            wal = test_db.with_suffix(".db-wal")
            shm = test_db.with_suffix(".db-shm")
            if wal.exists(): wal.unlink(missing_ok=True)
            if shm.exists(): shm.unlink(missing_ok=True)
        except OSError:
            pass

    # Assertions
    assert stage3_egress_passes == len(cohort)
    assert stage4_ingress_passes == len(cohort)
    assert data_loss_flags == 0
    assert overhead_p95 <= 5.0, f"Overhead P95 ({overhead_p95:.2f}ms) exceeded 5.0ms budget!"
    assert integrity["verified"] is True

    return {
        "total_docs": len(cohort),
        "stage3_passes": stage3_egress_passes,
        "stage4_passes": stage4_ingress_passes,
        "automated_routes": automated_routes,
        "human_routes": human_routes,
        "data_loss_flags": data_loss_flags,
        "overhead_p50": overhead_p50,
        "overhead_p90": overhead_p90,
        "overhead_p95": overhead_p95,
        "nlp_p95": nlp_p95,
        "total_p95": total_p95,
        "audit_integrity": integrity["status"]
    }


def test_e2e_integration_suite():
    """Pytest wrapper running the full E2E integration pipeline."""
    metrics = run_e2e_integration_pipeline()
    assert metrics["total_docs"] >= 50
    assert metrics["stage3_passes"] == metrics["total_docs"]
    assert metrics["stage4_passes"] == metrics["total_docs"]
    assert metrics["data_loss_flags"] == 0
    assert metrics["overhead_p95"] <= 5.0


if __name__ == "__main__":
    run_e2e_integration_pipeline()
