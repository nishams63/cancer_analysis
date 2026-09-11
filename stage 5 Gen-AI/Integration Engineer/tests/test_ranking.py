from src.ranking import WildcardRanker, CandidateFilter

def test_wildcard_ranking_scoring():
    rec1 = {
        "scenario_id": "SYN-S001",
        "batch_id": "B-001",
        "status": "SUCCESS",
        "evaluation": {
            "scenario_plausibility": {"score": 0.95},
            "fidelity": {"fidelity_score": 1.0},
            "narrative_faithfulness": {"faithfulness_score": 0.95},
            "difficulty": {"difficulty_score": 85.0},
            "impact": {"impact_score": 90.0}
        },
        "stages": {"stage1": {}, "stage4": {}},
        "failures": [{"stage": "stage1", "code": "F03"}, {"stage": "stage4", "code": "F06"}],
        "counterfactuals": [{"instability_detected": True}]
    }

    ranker = WildcardRanker()
    ranked = ranker.rank_candidates([rec1])
    assert len(ranked) == 1
    assert ranked[0]["rank"] == 1
    assert ranked[0]["status_label"] == "CANDIDATE ONLY"
    assert "Scenario Plausibility" in ranked[0]["reason_explanation"]

def test_candidate_filtering_ineligible():
    cf = CandidateFilter(min_fidelity=0.80, min_faithfulness=0.80)
    bad_rec = {
        "scenario_id": "SYN-BAD",
        "batch_id": "B-001",
        "status": "SUCCESS",
        "evaluation": {
            "fidelity": {"fidelity_score": 0.50},  # Below 0.80 threshold
            "narrative_faithfulness": {"faithfulness_score": 0.90}
        }
    }
    ok, reason = cf.is_eligible(bad_rec)
    assert ok is False
    assert "Fidelity" in reason
