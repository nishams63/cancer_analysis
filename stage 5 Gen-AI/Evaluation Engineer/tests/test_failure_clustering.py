"""Tests for Failure Clustering and Wildcard Evidence Builder."""
import pytest
from src.evaluation.failure_clustering import FailureClusterer
from src.evaluation.wildcard_evidence import WildcardEvidenceBuilder


def test_failure_clustering():
    clusterer = FailureClusterer()
    failures = [
        {"failure_code": "F06", "stage": "stage4", "scenario_id": "S1", "confidence": 0.92, "difficulty_score": 75.0},
        {"failure_code": "F06", "stage": "stage4", "scenario_id": "S2", "confidence": 0.90, "difficulty_score": 80.0},
        {"failure_code": "F01", "stage": "stage3", "scenario_id": "S1", "confidence": 0.88, "difficulty_score": 75.0},
    ]
    df = clusterer.cluster_failures(failures)
    assert len(df) == 2 # 2 distinct (code, stage) pairs
    assert "CLUST-001" in df["cluster_id"].values


def test_wildcard_candidate_table():
    builder = WildcardEvidenceBuilder()
    dossiers = [
        {"scenario_id": "S1", "difficulty_score": 85.0, "impact_score": 90.0, "scenario_plausibility_score": 0.95, "narrative_faithfulness_score": 0.98, "failed_stages": ["stage3", "stage4"], "all_failure_codes": ["F01", "F06"]},
        {"scenario_id": "S2", "difficulty_score": 45.0, "impact_score": 50.0, "scenario_plausibility_score": 0.90, "narrative_faithfulness_score": 0.90, "failed_stages": ["stage1"], "all_failure_codes": ["F03"]}
    ]
    df = builder.build_candidate_table(dossiers)
    assert len(df) == 2
    assert df.iloc[0]["scenario_id"] == "S1" # Higher evidence score ranked #1
    assert df.iloc[0]["evidence_score"] > df.iloc[1]["evidence_score"]
