"""Pytest fixtures for Evaluation Engineer."""
import sys
from pathlib import Path
import pytest

ROLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROLE_ROOT))
sys.path.insert(0, str(ROLE_ROOT / "src"))

@pytest.fixture
def sample_valid_patient():
    return {
        "scenario_id": "PROMPT-R01",
        "patient_id": "PT-SYN-0001",
        "demographics": {"age": 62, "sex": "Female", "cancer_type": "NSCLC"},
        "mutations": ["EGFR L858R", "EGFR T790M", "MET Amplification"],
        "biomarkers": {"tumor_size_cm": 4.2, "creatinine_level": 1.1},
        "treatments": [{"treatment_name": "Osimertinib 80mg", "line": 2}],
        "timeline": [{"event": "Diagnosis", "day": 0}, {"event": "Progression", "day": 180}]
    }

@pytest.fixture
def sample_scenario_def():
    return {
        "scenario_id": "PROMPT-R01",
        "patient_skeleton": {"age": 62, "sex": "Female", "cancer_type": "NSCLC"},
        "required_entities": [
            {"category": "mutation", "values": ["EGFR", "T790M"]},
            {"category": "mutation", "values": ["MET"]},
            {"category": "treatment", "values": ["Osimertinib"]}
        ],
        "forbidden_modifications": [
            {"rule_id": "FORBID-01", "forbidden_pattern": "MET negative"}
        ],
        "evidence_citations": ["NCCN-NSCLC-v4.2024"]
    }
