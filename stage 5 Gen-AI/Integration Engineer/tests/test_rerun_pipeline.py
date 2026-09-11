from src.integration.orchestrator import MasterOrchestrator
from src.storage.failure_store import FailureStore

def test_rerun_preserves_parent(temp_db):
    orch = MasterOrchestrator(db_path=temp_db)
    # 1. Run parent batch
    p_res = orch.run_batch(n=4, seed=42, batch_id="PARENT-001")
    assert p_res["status"] == "COMPLETED"

    # 2. Run child rerun batch
    c_res = orch.run_batch(n=2, seed=99, batch_id="RERUN-001")
    assert c_res["status"] == "COMPLETED"

    # Verify both batches exist independently
    b_parent = orch.batch_store.get_batch("PARENT-001")
    b_child = orch.batch_store.get_batch("RERUN-001")
    assert b_parent is not None
    assert b_child is not None
    assert b_parent["batch_id"] != b_child["batch_id"]
