"""Tests for Blind-Spot Taxonomy, Ranking, and Cross-Stage Failure Analyzers."""
import pytest
import pandas as pd
from src.eda import (
    BlindSpotRanker, FailurePatternAnalyzer, DisagreementAnalyzer,
    SparseRegionAnalyzer, MutationRarityAnalyzer
)


def test_all_15_blind_spots_present():
    ranker = BlindSpotRanker()
    df = ranker.rank_blind_spots()
    assert len(df) == 15
    expected_ids = {f"BS{i:02d}" for i in range(1, 16)}
    actual_ids = set(df["blind_spot_id"])
    assert actual_ids == expected_ids


def test_priority_scores_bounded():
    ranker = BlindSpotRanker()
    df = ranker.rank_blind_spots()
    assert (df["stress_test_priority_score"] >= 0.0).all()
    assert (df["stress_test_priority_score"] <= 1.0).all()
    # Ensure descending order
    scores = df["stress_test_priority_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_priority_tiers_valid():
    ranker = BlindSpotRanker()
    df = ranker.rank_blind_spots()
    valid_tiers = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
    assert set(df["priority_tier"]).issubset(valid_tiers)
    assert "CRITICAL" in set(df["priority_tier"])


def test_failure_patterns_exist():
    fa = FailurePatternAnalyzer()
    df = fa.analyze()
    assert len(df) >= 6
    assert "affected_stage" in df.columns
    assert "observed_error_rate" in df.columns
    assert (df["observed_error_rate"] > 0.0).all()


def test_disagreement_patterns_exist():
    da = DisagreementAnalyzer()
    df = da.analyze()
    assert len(df) >= 3
    assert "disagreement_type" in df.columns
    assert "candidate_scenario" in df.columns