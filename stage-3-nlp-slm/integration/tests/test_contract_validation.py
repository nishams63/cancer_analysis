"""
Unit tests for Stage 3 -> Stage 4 Contract Definition and Schema Validation.
"""

import pytest
import json
from datetime import datetime, timezone
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integration"))

from contract_validation import (
    Stage3OutputPayload,
    TriageUrgencyPayload,
    ToxicityHazardPayload,
    ClinicalEntityPayload,
    validate_stage3_output,
    validate_stage4_input,
    export_canonical_json,
    ContractValidationError,
    SUPPORTED_SCHEMA_VERSION,
)


def make_valid_payload_dict():
    return {
        "schema_version": "1.0.0",
        "document_id": "DOC-VALID-001",
        "patient_id": "PT-009123",
        "inference_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": "stage3-minilm-hybrid-v3.3-config-c",
        "ner_model_version": "stage3-trainable-ner-v1.0",
        "triage_urgency": {
            "predicted_class": "CRITICAL",
            "confidence": 0.9850,
            "class_probabilities": {
                "LOW": 0.0050,
                "MEDIUM": 0.0050,
                "HIGH": 0.0050,
                "CRITICAL": 0.9850
            }
        },
        "toxicity_hazard": {
            "predicted_class": "CARDIAC",
            "confidence": 0.9420,
            "class_probabilities": {
                "NONE": 0.0100,
                "HEMATOLOGIC": 0.0100,
                "HEPATIC": 0.0100,
                "RENAL": 0.0100,
                "CARDIAC": 0.9420,
                "PULMONARY": 0.0080,
                "NEUROPATHIC": 0.0050,
                "DERMATOLOGIC": 0.0050
            }
        },
        "clinical_entities": [
            {
                "start": 10,
                "end": 22,
                "label": "DRUG_NAME",
                "text": "trastuzumab",
                "polarity": "AFFIRMED",
                "confidence": 0.9980
            },
            {
                "start": 30,
                "end": 43,
                "label": "ADVERSE_EVENT",
                "text": "heart failure",
                "polarity": "AFFIRMED",
                "confidence": 0.9250
            }
        ],
        "raw_text_hash": "a" * 64,
        "integration_metadata": {"test_env": "unit"}
    }


def test_valid_payload_validation():
    raw = make_valid_payload_dict()
    obj3 = validate_stage3_output(raw)
    assert obj3.document_id == "DOC-VALID-001"
    assert obj3.triage_urgency.predicted_class == "CRITICAL"
    assert len(obj3.clinical_entities) == 2

    # Verify Stage 4 boundary also validates
    obj4 = validate_stage4_input(obj3)
    assert obj4.patient_id == "PT-009123"

    # Export canonical JSON
    canon = export_canonical_json(obj3)
    assert isinstance(canon, str)
    loaded = json.loads(canon)
    assert loaded["schema_version"] == SUPPORTED_SCHEMA_VERSION


def test_missing_required_field():
    raw = make_valid_payload_dict()
    del raw["patient_id"]
    with pytest.raises(ContractValidationError) as exc:
        validate_stage3_output(raw)
    assert "patient_id" in str(exc.value)


def test_invalid_urgency_class():
    raw = make_valid_payload_dict()
    raw["triage_urgency"]["predicted_class"] = "EXTREME_EMERGENCY"
    with pytest.raises(ContractValidationError):
        validate_stage3_output(raw)


def test_invalid_hazard_class():
    raw = make_valid_payload_dict()
    raw["toxicity_hazard"]["predicted_class"] = "UNKNOWN_TOXICITY"
    with pytest.raises(ContractValidationError):
        validate_stage3_output(raw)


def test_confidence_out_of_bounds():
    raw = make_valid_payload_dict()
    raw["triage_urgency"]["confidence"] = 1.05
    with pytest.raises(ContractValidationError):
        validate_stage3_output(raw)

    raw2 = make_valid_payload_dict()
    raw2["toxicity_hazard"]["confidence"] = -0.01
    with pytest.raises(ContractValidationError):
        validate_stage3_output(raw2)


def test_invalid_entity_spans():
    raw = make_valid_payload_dict()
    # end <= start
    raw["clinical_entities"][0]["start"] = 25
    raw["clinical_entities"][0]["end"] = 20
    with pytest.raises(ContractValidationError):
        validate_stage3_output(raw)


def test_invalid_entity_label():
    raw = make_valid_payload_dict()
    raw["clinical_entities"][0]["label"] = "SYMPTOM_SEVERITY"
    with pytest.raises(ContractValidationError):
        validate_stage3_output(raw)


def test_invalid_schema_version():
    raw = make_valid_payload_dict()
    raw["schema_version"] = "v1"
    with pytest.raises(ContractValidationError):
        validate_stage3_output(raw)

    raw["schema_version"] = "2.0.0"  # Incompatible major version
    with pytest.raises(ContractValidationError):
        validate_stage3_output(raw)


def test_invalid_timestamp():
    raw = make_valid_payload_dict()
    raw["inference_timestamp"] = "yesterday afternoon"
    with pytest.raises(ContractValidationError):
        validate_stage3_output(raw)


def test_invalid_sha256_hash():
    raw = make_valid_payload_dict()
    raw["raw_text_hash"] = "invalid_hash_string"
    with pytest.raises(ContractValidationError):
        validate_stage3_output(raw)


def test_extra_forbidden_fields():
    raw = make_valid_payload_dict()
    raw["unauthorized_clinical_note_text"] = "should fail extra forbidden check"
    with pytest.raises(ContractValidationError):
        validate_stage3_output(raw)


def test_non_dict_payload_rejection():
    with pytest.raises(ContractValidationError) as exc:
        validate_stage3_output("not a dict")
    assert exc.value.raw_payload == "not a dict"

    with pytest.raises(ContractValidationError):
        validate_stage4_input([1, 2, 3])


def test_stage4_input_dict_validation():
    raw = make_valid_payload_dict()
    obj = validate_stage4_input(raw)
    assert obj.document_id == "DOC-VALID-001"


def test_class_probabilities_validation_errors():
    raw = make_valid_payload_dict()
    raw["triage_urgency"]["class_probabilities"] = {"INVALID_CLASS": 0.5}
    with pytest.raises(ContractValidationError):
        validate_stage3_output(raw)

    raw2 = make_valid_payload_dict()
    raw2["triage_urgency"]["class_probabilities"] = {"LOW": 1.5}
    with pytest.raises(ContractValidationError):
        validate_stage3_output(raw2)

    raw3 = make_valid_payload_dict()
    raw3["toxicity_hazard"]["class_probabilities"] = {"INVALID_HAZARD": 0.5}
    with pytest.raises(ContractValidationError):
        validate_stage3_output(raw3)

    raw4 = make_valid_payload_dict()
    raw4["toxicity_hazard"]["class_probabilities"] = {"RENAL": -0.2}
    with pytest.raises(ContractValidationError):
        validate_stage3_output(raw4)

