import pytest
from src.evaluator import StressTestEvaluator
from src.metrics import calculate_sensitivity_drop, calculate_discordance_rate


def test_evaluator_catches_vulnerability():
    ev = StressTestEvaluator()
    sc = {"scenario_id": "PROMPT-R01", "target_blind_spots": ["BS01"]}
    pred = {"risk_category": "Standard", "contraindication_detected": False}
    res = ev.evaluate_scenario(sc, pred)
    assert res["is_vulnerability_exposed"] is True


def test_sensitivity_drop_calculation():
    drop = calculate_sensitivity_drop(0.92, 0.65)
    assert drop == 0.27
