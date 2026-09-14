"""Tests for multi-factor conflict resolution between competing hypotheses."""
from reasoning.conflict_resolution import resolve_competing_causes


def test_clear_winner_resolution():
    candidates = [
        {"cause": "Price Hike", "effect_size": 0.85, "sample_size": 10000, "evidence_strength": 0.90, "consistency": 0.85},
        {"cause": "Competitor Launch", "effect_size": 0.40, "sample_size": 10000, "evidence_strength": 0.45, "consistency": 0.50},
    ]
    winner, margin, needs_esc = resolve_competing_causes(candidates)
    assert winner["cause"] == "Price Hike"
    assert margin > 0.10
    assert needs_esc is False


def test_ambiguous_contest_triggers_escalation():
    candidates = [
        {"cause": "Hypothesis A", "effect_size": 0.70, "sample_size": 5000, "evidence_strength": 0.70, "consistency": 0.70},
        {"cause": "Hypothesis B", "effect_size": 0.68, "sample_size": 5000, "evidence_strength": 0.70, "consistency": 0.69},
    ]
    winner, margin, needs_esc = resolve_competing_causes(candidates)
    assert margin < 0.10
    assert needs_esc is True


def test_single_hypothesis():
    candidates = [{"cause": "Lone Cause", "effect_size": 0.8}]
    winner, margin, needs_esc = resolve_competing_causes(candidates)
    assert winner["cause"] == "Lone Cause"
    assert needs_esc is False
