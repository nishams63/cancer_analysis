from src.integration.orchestrator import MasterOrchestrator
from src.integration.batch_manager import BatchState

def test_orchestrator_run_batch(temp_db):
    orchestrator = MasterOrchestrator(db_path=temp_db)
    res = orchestrator.run_batch(n=3, seed=42)
    assert res["status"] == "COMPLETED"
    assert res["scenario_count"] == 3
    assert res["ranked_candidates_count"] == 3

def test_orchestrator_scenario_filter(temp_db):
    orchestrator = MasterOrchestrator(db_path=temp_db)
    res = orchestrator.run_batch(n=2, seed=123, scenario_id_filter="PROMPT-R01")
    assert res["status"] == "COMPLETED"
    assert res["scenario_count"] == 2
