"""Unit tests for Narrative Validator ensuring language preserves structured facts."""
import pytest
from src.validation.narrative_validator import NarrativeValidator


@pytest.fixture
def patient_and_scenario():
    patient = {
        "demographics": {"age": 62, "sex": "Female", "cancer_type": "NSCLC"},
        "mutations": ["EGFR L858R", "EGFR T790M"],
        "treatments": [{"treatment_name": "Osimertinib 80mg daily"}],
        "dosages": {"Osimertinib": 80.0},
        "missing_fields": ["tumor_size_cm"],
        "biomarkers": {}
    }
    scenario = {
        "scenario_id": "PROMPT-R01",
        "forbidden_modifications": [
            {"forbidden_pattern": "treatment-naive|first-line therapy", "description": "Must not be treatment naive"}
        ]
    }
    return patient, scenario


def test_factual_narrative_passes(patient_and_scenario):
    patient, scenario = patient_and_scenario
    narrative = (
        "Patient is a 62-year-old female with NSCLC. Staging shows metastatic disease. "
        "Molecular testing demonstrates EGFR L858R and acquired EGFR T790M mutation. "
        "She is maintained on Osimertinib 80mg daily. Baseline primary tumor size is undocumented."
    )
    nv = NarrativeValidator()
    res = nv.validate(narrative, patient, scenario)
    assert res.valid is True
    assert res.checks["mutations"] == "PASS"
    assert res.checks["treatment"] == "PASS"
    assert res.checks["missingness"] == "PASS"


def test_dropped_mutation_fails(patient_and_scenario):
    patient, scenario = patient_and_scenario
    # Omit T790M
    narrative = (
        "Patient is a 62-year-old female with NSCLC. Testing demonstrates EGFR L858R. "
        "Receives Osimertinib 80mg daily."
    )
    nv = NarrativeValidator()
    res = nv.validate(narrative, patient, scenario)
    assert res.valid is False
    assert res.checks["mutations"] == "FAIL"
    assert any("EGFR T790M" in v["reason"] for v in res.violations)


def test_hallucinated_missing_field_fails(patient_and_scenario):
    patient, scenario = patient_and_scenario
    # Hallucinates tumor_size_cm measurement
    narrative = (
        "Patient is a 62-year-old female with NSCLC, EGFR L858R and EGFR T790M. "
        "Receives Osimertinib 80mg daily. On imaging, primary tumor measures 4.2 cm."
    )
    nv = NarrativeValidator()
    res = nv.validate(narrative, patient, scenario)
    assert res.valid is False
    assert res.checks["missingness"] == "FAIL"
    assert any("missing field" in v["reason"] for v in res.violations)


def test_forbidden_pattern_fails(patient_and_scenario):
    patient, scenario = patient_and_scenario
    # Claims treatment-naive
    narrative = (
        "Patient is a 62-year-old female with NSCLC, EGFR L858R, EGFR T790M, on Osimertinib. "
        "She presents as a treatment-naive individual."
    )
    nv = NarrativeValidator()
    res = nv.validate(narrative, patient, scenario)
    assert res.valid is False
    assert res.checks["forbidden_modifications"] == "FAIL"
