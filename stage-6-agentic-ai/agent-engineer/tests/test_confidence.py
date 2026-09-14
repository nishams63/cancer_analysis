"""Tests for deterministic confidence score derivation."""
import pytest
from reasoning.confidence import calculate_confidence


def test_standard_confidence():
    score = calculate_confidence(
        data_quality_score=0.90,
        p_value=0.001,
        sample_size=5000,
        cause_margin=0.20,
    )
    assert 0.80 <= score <= 1.0


def test_confidence_with_critical_anomaly():
    score_clean = calculate_confidence(data_quality_score=0.90, has_unaddressed_critical_anomaly=False)
    score_anom = calculate_confidence(data_quality_score=0.90, has_unaddressed_critical_anomaly=True)
    assert score_clean - score_anom == pytest.approx(0.15, abs=1e-3)


def test_confidence_clamping():
    # Extreme low inputs
    score_low = calculate_confidence(
        data_quality_score=0.0,
        p_value=0.99,
        sample_size=10,
        cause_margin=0.0,
        has_unaddressed_critical_anomaly=True,
    )
    assert score_low >= 0.0

    # Extreme high inputs
    score_high = calculate_confidence(
        data_quality_score=1.0,
        p_value=0.00001,
        sample_size=50000,
        cause_margin=0.50,
        has_unaddressed_critical_anomaly=False,
    )
    assert score_high <= 1.0
