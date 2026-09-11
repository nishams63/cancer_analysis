"""Tests for Scenario Builder and Catalog Generation."""
import pytest
from src.prompts.scenario_builder import get_core_scenarios, build_scenario_catalog


def test_all_core_scenarios_built():
    scenarios = get_core_scenarios()
    assert len(scenarios) >= 15
    ids = [s.scenario_id for s in scenarios]
    assert len(ids) == len(set(ids)), "Scenario IDs must be unique"
    for i in range(1, 16):
        assert f"PROMPT-R{i:02d}" in ids


def test_scenario_fields_populated():
    scenarios = get_core_scenarios()
    for sc in scenarios:
        assert sc.title, f"{sc.scenario_id} missing title"
        assert sc.clinical_premise, f"{sc.scenario_id} missing premise"
        assert sc.target_stage_vulnerability, f"{sc.scenario_id} missing vulnerability"
        assert sc.prompt_template, f"{sc.scenario_id} missing prompt template"
        assert len(sc.required_entities) > 0, f"{sc.scenario_id} missing required entities"
        assert sc.rag_intent is not None, f"{sc.scenario_id} missing RAG intent"


def test_patient_skeleton_valid():
    scenarios = get_core_scenarios()
    for sc in scenarios:
        sk = sc.patient_skeleton
        assert sk.age > 0 and sk.age < 120
        assert sk.sex in {"M", "F"}
        assert sk.cancer_type
        assert 0 <= sk.ecog_ps <= 4


def test_build_scenario_catalog():
    catalog = build_scenario_catalog()
    assert catalog.total_scenarios == len(catalog.scenarios)
    assert catalog.version == "1.0.0"