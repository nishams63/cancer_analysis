"""
Unit tests for Failure Mode Handlers (failure_mode_handlers.py).
Asserts that every failure mode degrades safely to HUMAN_REVIEW (never auto-approved).
"""

import pytest
import time
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integration"))

from failure_mode_handlers import (
    IntegrationFailureHandler,
    Stage3TimeoutError,
    MalformedDocumentError,
    ModelVersionMismatchError,
    PartialPipelineError,
    DegradedEnvelope,
)
from confidence_gate import ConfidenceGate


def make_sample_raw_output(empty_entities=False, model_version="stage3-minilm-hybrid-v3.3", ner_version="stage3-ner-v1.0"):
    entities = [] if empty_entities else [
        {
            "start": 0,
            "end": 9,
            "label": "DRUG_NAME",
            "text": "erlotinib",
            "polarity": "AFFIRMED",
            "confidence": 0.99
        }
    ]
    return {
        "schema_version": "1.0.0",
        "document_id": "DOC-FAIL-TEST",
        "patient_id": "PT-FAIL-001",
        "inference_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "ner_model_version": ner_version,
        "triage_urgency": {
            "predicted_class": "HIGH",
            "confidence": 0.88,
            "class_probabilities": {"LOW": 0.02, "MEDIUM": 0.10, "HIGH": 0.88, "CRITICAL": 0.00}
        },
        "toxicity_hazard": {
            "predicted_class": "NONE",
            "confidence": 0.92,
            "class_probabilities": {"NONE": 0.92}
        },
        "clinical_entities": entities
    }


def test_input_sanitization_success():
    handler = IntegrationFailureHandler()
    res = handler.sanitize_input_document("Patient presents with NSCLC and cough.", "D1", "P1")
    assert res == "Patient presents with NSCLC and cough."


def test_input_sanitization_malformed_failures():
    handler = IntegrationFailureHandler()
    with pytest.raises(MalformedDocumentError):
        handler.sanitize_input_document(None, "D1", "P1")
    with pytest.raises(MalformedDocumentError):
        handler.sanitize_input_document(12345, "D1", "P1")
    with pytest.raises(MalformedDocumentError):
        handler.sanitize_input_document("   ", "D1", "P1")
    with pytest.raises(MalformedDocumentError):
        handler.sanitize_input_document("abc", "D1", "P1")  # Below length floor
    with pytest.raises(MalformedDocumentError):
        handler.sanitize_input_document("Text with \x00 null byte", "D1", "P1")


def test_timeout_immediate_escalation():
    # Set short timeout of 20ms
    handler = IntegrationFailureHandler(timeout_seconds=0.020)

    def slow_inference():
        time.sleep(0.035)
        return make_sample_raw_output()

    envelope = handler.handle_safely("DOC-SLOW", "PT-001", slow_inference)
    assert envelope.status == "TIMEOUT"
    assert envelope.destination == "HUMAN_REVIEW"
    assert envelope.is_degraded is True
    assert "Immediate human review escalation triggered" in str(envelope.error_details)


def test_empty_clinical_entities_safety_flag():
    handler = IntegrationFailureHandler(require_non_empty_entities=True)

    def empty_ner_inference():
        return make_sample_raw_output(empty_entities=True)

    envelope = handler.handle_safely("DOC-EMPTY-ENT", "PT-001", empty_ner_inference)
    assert envelope.status == "EMPTY_ENTITIES"
    assert envelope.destination == "HUMAN_REVIEW"
    assert "Empty clinical entities extracted" in envelope.degradation_reason


def test_model_version_mismatch():
    handler = IntegrationFailureHandler(expected_model_prefix="stage3-")

    # Wrong model version
    def wrong_model():
        return make_sample_raw_output(model_version="legacy-tfidf-v1.0")

    env1 = handler.handle_safely("DOC-VER-1", "PT-001", wrong_model)
    assert env1.status == "VERSION_MISMATCH"
    assert env1.destination == "HUMAN_REVIEW"

    # Wrong NER version
    def wrong_ner():
        return make_sample_raw_output(ner_version="uncalibrated-spacy-v0.1")

    env2 = handler.handle_safely("DOC-VER-2", "PT-001", wrong_ner)
    assert env2.status == "VERSION_MISMATCH"
    assert env2.destination == "HUMAN_REVIEW"


def test_partial_pipeline_failures():
    handler = IntegrationFailureHandler()

    # Triage missing, entities present
    def partial_missing_triage():
        out = make_sample_raw_output()
        del out["triage_urgency"]
        return out

    env1 = handler.handle_safely("DOC-PARTIAL-1", "PT-001", partial_missing_triage)
    assert env1.status == "PARTIAL_FAILURE"
    assert env1.destination == "HUMAN_REVIEW"

    # Entities missing, triage present
    def partial_missing_ner():
        out = make_sample_raw_output()
        del out["clinical_entities"]
        return out

    env2 = handler.handle_safely("DOC-PARTIAL-2", "PT-001", partial_missing_ner)
    assert env2.status == "PARTIAL_FAILURE"
    assert env2.destination == "HUMAN_REVIEW"


def test_contract_validation_rejection_inside_handler():
    handler = IntegrationFailureHandler()

    def invalid_contract_output():
        out = make_sample_raw_output()
        out["triage_urgency"]["confidence"] = 1.99  # Invalid confidence
        return out

    env = handler.handle_safely("DOC-CORRUPT", "PT-001", invalid_contract_output)
    assert env.status == "MALFORMED_INPUT"
    assert env.destination == "HUMAN_REVIEW"


def test_unexpected_exception_internal_error():
    handler = IntegrationFailureHandler()

    def crash_inference():
        raise RuntimeError("Memory segmentation fault simulation")

    env = handler.handle_safely("DOC-CRASH", "PT-001", crash_inference)
    assert env.status == "INTERNAL_ERROR"
    assert env.destination == "HUMAN_REVIEW"
    assert "Memory segmentation fault" in str(env.error_details)


def test_healthy_execution_success():
    handler = IntegrationFailureHandler()

    def good_inference():
        return make_sample_raw_output()

    env = handler.handle_safely("DOC-HEALTHY", "PT-001", good_inference)
    assert env.status == "SUCCESS"
    assert env.destination == "STAGE_4_AUTOMATED"
    assert env.is_degraded is False
    assert env.payload is not None
    assert env.routing_result is not None


def test_empty_clinical_entities_allowed_when_flag_false():
    handler = IntegrationFailureHandler(require_non_empty_entities=False)

    def empty_ner_inference():
        return make_sample_raw_output(empty_entities=True)

    env = handler.handle_safely("DOC-EMPTY-ALLOWED", "PT-001", empty_ner_inference)
    assert env.status == "SUCCESS"
    assert env.destination == "STAGE_4_AUTOMATED"
    assert len(env.payload.clinical_entities) == 0

