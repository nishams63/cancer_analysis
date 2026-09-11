from src.adapters import Stage1Adapter, Stage2Adapter, Stage3Adapter, Stage4Adapter, StageStatus

def test_stage1_adapter(mock_scenario_data):
    adapter = Stage1Adapter()
    res = adapter.run("SYN-001", mock_scenario_data)
    assert res.status == StageStatus.SUCCESS.value
    assert "hazard_score" in res.prediction
    assert res.latency_ms >= 0.0

def test_stage2_adapter_missing_imaging(mock_scenario_data):
    adapter = Stage2Adapter()
    res = adapter.run("SYN-001", mock_scenario_data)
    # Per architectural rule: do not fabricate images!
    assert res.status == StageStatus.SKIPPED_INPUT_UNAVAILABLE.value

def test_stage3_adapter_nlp(mock_scenario_data):
    adapter = Stage3Adapter()
    res = adapter.run("SYN-001", mock_scenario_data)
    assert res.status == StageStatus.SUCCESS.value
    assert "extracted_entities" in res.prediction
    assert res.prediction["triage_urgency"] == "URGENT"

def test_stage4_adapter_recommendation(mock_scenario_data):
    adapter = Stage4Adapter()
    res = adapter.run("SYN-001", mock_scenario_data)
    assert res.status == StageStatus.SUCCESS.value
    assert "Osimertinib + Savolitinib" in res.prediction["recommended_regimen"]
