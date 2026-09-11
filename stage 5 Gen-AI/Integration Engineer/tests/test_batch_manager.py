from src.integration.batch_manager import BatchManager, BatchState
from src.storage.batch_store import BatchStore

def test_batch_lifecycle(result_store):
    bs = BatchStore(result_store)
    bm = BatchManager(bs)
    bm.start_batch("B-TEST-100", 42, 10, {"seed": 42}, {"version": "1.0"})
    
    b1 = bs.get_batch("B-TEST-100")
    assert b1["status"] == BatchState.CREATED.value

    bm.transition("B-TEST-100", BatchState.VALIDATING)
    assert bs.get_batch("B-TEST-100")["status"] == BatchState.VALIDATING.value

    bm.transition("B-TEST-100", BatchState.COMPLETED, actual_count=10)
    b_fin = bs.get_batch("B-TEST-100")
    assert b_fin["status"] == BatchState.COMPLETED.value
    assert b_fin["actual_count"] == 10
