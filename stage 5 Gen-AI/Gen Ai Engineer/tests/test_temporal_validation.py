"""Unit tests for temporal causality and timeline validation."""
import pytest
from src.validation.temporal_validator import TemporalValidator


def test_valid_timeline_passes():
    tv = TemporalValidator()
    patient = {
        "timeline": [
            {"event_type": "diagnosis", "date": "2024-01-10", "delta_days": 0},
            {"event_type": "treatment", "date": "2024-02-01", "delta_days": 22},
            {"event_type": "progression", "date": "2024-08-15", "delta_days": 218}
        ]
    }
    is_valid, errors = tv.validate(patient)
    assert is_valid is True
    assert len(errors) == 0


def test_anachronistic_timeline_fails():
    tv = TemporalValidator()
    patient = {
        "timeline": [
            {"event_type": "diagnosis", "date": "2024-08-15", "delta_days": 218},
            {"event_type": "treatment", "date": "2024-02-01", "delta_days": 22}
        ]
    }
    is_valid, errors = tv.validate(patient)
    assert is_valid is False
    assert any("Chronological regression" in e for e in errors)


def test_treatment_before_diagnosis_fails():
    tv = TemporalValidator()
    patient = {
        "timeline": [
            {"event_type": "treatment", "date": "2023-12-01", "delta_days": 0},
            {"event_type": "progression", "date": "2024-01-15", "delta_days": 45}
        ]
    }
    is_valid, errors = tv.validate(patient)
    assert is_valid is False
    assert any("First timeline event must be diagnosis" in e for e in errors)
