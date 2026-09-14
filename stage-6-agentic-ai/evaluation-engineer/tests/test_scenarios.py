"""Tests for EvaluationScenario schema and loaded scenario catalog."""
from pathlib import Path
import json
from schemas.scenario import EvaluationScenario, ScenarioCategory


def test_scenario_catalog_loading(runner):
    scenarios = runner._scenarios
    assert len(scenarios) >= 12
    assert "SC-REV-001" in scenarios
    assert "SC-EDGE-001" in scenarios
    assert "SC-EDGE-003" in scenarios


def test_scenario_fields_validation(sample_scenario):
    assert sample_scenario.scenario_id == "SC-REV-001"
    assert sample_scenario.category == ScenarioCategory.SALES
    assert len(sample_scenario.expected_tasks) == 4
    assert len(sample_scenario.expected_tools) == 4
