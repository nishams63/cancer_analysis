"""Unit tests for Structured Synthetic Patient Sampler."""
import pytest
from src.generation import ScenarioLoader, StructuredSampler
from src.generation.patient_builder import StructuredSyntheticPatient


@pytest.fixture
def scenario():
    loader = ScenarioLoader()
    return loader.get_scenario("PROMPT-R01")


def test_sampler_generates_valid_patient(scenario):
    sampler = StructuredSampler()
    pt = sampler.sample_patient(scenario, seed=123)
    assert isinstance(pt, StructuredSyntheticPatient)
    assert pt.synthetic is True
    assert pt.scenario_id == "PROMPT-R01"
    assert pt.patient_id.startswith("SYN-P")
    assert pt.demographics.cancer_type == "NSCLC"
    assert len(pt.mutations) > 0
    assert len(pt.timeline) >= 2


def test_sampler_reproducibility(scenario):
    sampler1 = StructuredSampler()
    sampler2 = StructuredSampler()
    pt1 = sampler1.sample_patient(scenario, seed=999)
    pt2 = sampler2.sample_patient(scenario, seed=999)
    assert pt1.to_dict()["demographics"] == pt2.to_dict()["demographics"]
    assert pt1.to_dict()["mutations"] == pt2.to_dict()["mutations"]
    assert pt1.to_dict()["biomarkers"] == pt2.to_dict()["biomarkers"]


def test_sampler_distinct_seeds_produce_variation(scenario):
    sampler = StructuredSampler()
    pt1 = sampler.sample_patient(scenario, seed=101)
    pt2 = sampler.sample_patient(scenario, seed=202)
    # At least one biomarker or mutation set differs
    d1 = pt1.to_dict()
    d2 = pt2.to_dict()
    assert (d1["biomarkers"] != d2["biomarkers"]) or (d1["mutations"] != d2["mutations"]) or (d1["demographics"]["age"] != d2["demographics"]["age"])


def test_sampler_enforces_required_entities(scenario):
    sampler = StructuredSampler()
    pt = sampler.sample_patient(scenario, seed=42)
    p_dict = pt.to_dict()
    muts = p_dict["mutations"]
    all_muts_str = " ".join(muts)
    # PROMPT-R01 requires EGFR T790M or EGFR L858R + T790M
    assert "T790M" in all_muts_str or "L858R" in all_muts_str
