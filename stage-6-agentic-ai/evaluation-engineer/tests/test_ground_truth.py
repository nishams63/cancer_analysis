"""Tests for GroundTruth schema and tolerance intervals."""
from schemas.ground_truth import GroundTruth, ExpectedFact


def test_ground_truth_registry_loading(runner):
    gts = runner._ground_truths
    assert len(gts) >= 12
    assert "GT-REV-001" in gts
    assert "GT-EDGE-003" in gts


def test_expected_facts_tolerances(sample_ground_truth):
    facts = sample_ground_truth.expected_facts
    assert len(facts) == 2
    f1 = facts[0]
    assert f1.metric_name == "decline_rate"
    assert f1.expected_value == 0.18
    assert f1.tolerance_pct == 0.05
    assert f1.is_critical is True
