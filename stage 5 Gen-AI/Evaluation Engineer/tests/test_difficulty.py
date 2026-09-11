"""Tests for Difficulty and Impact Calculators."""
import pytest
from src.evaluation.difficulty import DifficultyCalculator
from src.evaluation.impact import FailureImpactCalculator


def test_difficulty_increases_with_failures():
    calc = DifficultyCalculator()
    d_low = calc.calculate_difficulty(failure_count=0, has_disagreement=False, confidence_spread=0.05)
    d_high = calc.calculate_difficulty(failure_count=3, has_disagreement=True, confidence_spread=0.30)
    
    assert d_high["system_stress_test_difficulty_score"] > d_low["system_stress_test_difficulty_score"]
    assert d_high["difficulty_tier"] in ["HARD", "EXTREME"]


def test_difficulty_bounded_0_to_100():
    calc = DifficultyCalculator()
    res = calc.calculate_difficulty(failure_count=10, has_disagreement=True, confidence_spread=1.0, has_instability=True)
    assert 0.0 <= res["system_stress_test_difficulty_score"] <= 100.0


def test_failure_impact_critical_severity():
    impact_calc = FailureImpactCalculator()
    imp = impact_calc.calculate_impact(
        failure_codes=["F01", "F06"],
        failed_stages=["stage3", "stage4"],
        has_high_confidence_error=True
    )
    assert imp["failure_impact_score"] >= 70.0
    assert imp["severity_level"] in ["HIGH", "CRITICAL"]
