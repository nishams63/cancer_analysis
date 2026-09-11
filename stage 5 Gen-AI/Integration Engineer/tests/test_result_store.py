from src.storage import ResultStore, BatchStore, ScenarioStore, FailureStore

def test_result_store_crud(temp_db, mock_scenario_data):
    rs = ResultStore(temp_db)
    bs = BatchStore(rs)
    bs.create_batch("B-001", 42, 1, {"v": "1.0"}, {"v": "1.0"})

    rec = {
        "scenario_id": "SYN-S001",
        "batch_id": "B-001",
        "status": "SUCCESS",
        "scenario": mock_scenario_data["patient"],
        "rag": {"query": "test query", "retrieved_chunks": ["C-1"], "status": "SUCCESS", "concept_coverage": 1.0},
        "narrative": {"narrative_text": "sample note", "validation_status": "VALID"},
        "evaluation": {
            "realism": {"score": 0.85},
            "scenario_plausibility": {"score": 0.90},
            "rag_quality": {"score": 0.95},
            "fidelity": {"fidelity_score": 1.0},
            "narrative_faithfulness": {"faithfulness_score": 0.92},
            "difficulty": {"difficulty_score": 70.0},
            "impact": {"impact_score": 85.0}
        },
        "stages": {"stage1": {"status": "SUCCESS", "prediction": {}, "confidence": {}, "latency_ms": 10.0}},
        "failures": [{"stage": "stage1", "code": "F03", "confidence": 0.88, "evidence": "ctDNA risk error"}],
        "counterfactuals": []
    }
    rs.save_consolidated_record(rec)

    ss = ScenarioStore(rs)
    scen = ss.get_scenario("SYN-S001")
    assert scen is not None
    assert scen["fidelity_score"] == 1.0

    fs = FailureStore(rs)
    fails = fs.list_failures("B-001")
    assert len(fails) == 1
    assert fails[0]["failure_code"] == "F03"
