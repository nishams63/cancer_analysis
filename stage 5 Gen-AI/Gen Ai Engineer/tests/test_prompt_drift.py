"""Unit tests for prompt drift and scenario compliance."""
import pytest
from src.validation.prompt_drift_validator import PromptDriftValidator


def test_compliant_scenario_passes():
    pv = PromptDriftValidator()
    patient = {
        "demographics": {"age": 62},
        "mutations": ["EGFR L858R", "EGFR T790M"],
        "biomarkers": {"tumor_size_cm": 2.5},
        "missing_fields": []
    }
    scenario = {
        "patient_skeleton": {"age": 62},
        "required_entities": [
            {"name": "egfr_sensitizing", "entity_type": "mutation", "allowed_values": ["EGFR L858R"]},
            {"name": "egfr_resistance", "entity_type": "mutation", "allowed_values": ["EGFR T790M"]}
        ]
    }
    valid, violations, score = pv.validate(patient, scenario)
    assert valid is True
    assert len(violations) == 0
    assert score == 1.0


def test_drift_missing_mutation_detected():
    pv = PromptDriftValidator()
    patient = {
        "demographics": {"age": 62},
        "mutations": ["KRAS G12C"],
        "biomarkers": {},
        "missing_fields": []
    }
    scenario = {
        "patient_skeleton": {"age": 62},
        "required_entities": [
            {"name": "egfr_mutation", "entity_type": "mutation", "allowed_values": ["EGFR T790M"]}
        ]
    }
    valid, violations, score = pv.validate(patient, scenario)
    assert valid is False
    assert any("Required mutation" in v for v in violations)
    assert score < 1.0


def test_drift_age_shift_detected():
    pv = PromptDriftValidator()
    patient = {
        "demographics": {"age": 45},  # Shifted from skeleton age 68
        "mutations": [],
        "biomarkers": {},
        "missing_fields": []
    }
    scenario = {
        "patient_skeleton": {"age": 68},
        "required_entities": []
    }
    valid, violations, score = pv.validate(patient, scenario)
    assert valid is False
    assert any("Age shifted" in v for v in violations)
