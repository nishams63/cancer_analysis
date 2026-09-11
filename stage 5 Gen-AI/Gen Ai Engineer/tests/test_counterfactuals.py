"""Unit tests for single-variable counterfactual scenario generation."""
import pytest
from src.generation import ScenarioLoader, StructuredSampler, CounterfactualGenerator


def test_counterfactual_single_variable_mutation():
    loader = ScenarioLoader()
    scenario = loader.get_scenario("PROMPT-R05")
    sampler = StructuredSampler()
    factual_patient = sampler.sample_patient(scenario, seed=42)

    cfg = CounterfactualGenerator()
    cf_res = cfg.generate_counterfactual(
        factual_patient=factual_patient,
        target_variable="dosages",
        counterfactual_value={"Cisplatin": 0.0}
    )

    assert cf_res.target_variable == "dosages"
    assert cf_res.counterfactual_value == {"Cisplatin": 0.0}
    assert cf_res.counterfactual_patient.dosages == {"Cisplatin": 0.0}
    # Check that locked fields are identical
    assert cf_res.counterfactual_patient.demographics == factual_patient.demographics
    assert cf_res.counterfactual_patient.mutations == factual_patient.mutations
    assert cf_res.counterfactual_patient.biomarkers == factual_patient.biomarkers


def test_counterfactual_immutability_violation_caught():
    loader = ScenarioLoader()
    scenario = loader.get_scenario("PROMPT-R01")
    sampler = StructuredSampler()
    factual_patient = sampler.sample_patient(scenario, seed=42)

    cfg = CounterfactualGenerator()
    # Mutating treatments
    cf_res = cfg.generate_counterfactual(
        factual_patient=factual_patient,
        target_variable="treatments",
        counterfactual_value=[{"treatment_name": "Carboplatin + Pemetrexed", "line": 2}]
    )

    # All un-targeted fields must match
    f_dict = factual_patient.to_dict()
    cf_dict = cf_res.counterfactual_patient.to_dict()

    for k in ["demographics", "mutations", "biomarkers", "dosages", "adverse_events", "resistance", "timeline", "missing_fields"]:
        assert f_dict[k] == cf_dict[k], f"Field {k} unexpectedly modified in counterfactual!"


def test_counterfactual_serialization():
    loader = ScenarioLoader()
    scenario = loader.get_scenario("PROMPT-R01")
    sampler = StructuredSampler()
    factual_patient = sampler.sample_patient(scenario, seed=42)

    cfg = CounterfactualGenerator()
    cf_res = cfg.generate_counterfactual(
        factual_patient=factual_patient,
        target_variable="mutations",
        counterfactual_value=["EGFR L858R"]
    )
    d = cf_res.to_dict()
    assert "counterfactual_id" in d
    assert "factual_patient" in d
    assert "counterfactual_patient" in d
    assert "traceability" in d
    assert d["traceability"]["target_variable"] == "mutations"
