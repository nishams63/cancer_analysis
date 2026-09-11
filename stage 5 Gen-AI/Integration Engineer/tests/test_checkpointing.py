from src.integration.checkpoint_manager import CheckpointManager
import tempfile
import shutil

def test_checkpoint_save_and_load():
    tmp_dir = tempfile.mkdtemp()
    try:
        cm = CheckpointManager(tmp_dir)
        data = {"step": "generation", "status": "SUCCESS", "patient_id": "PT-001"}
        cm.save_checkpoint("B-001", "SYN-S001", "generation", data)

        assert cm.has_checkpoint("B-001", "SYN-S001", "generation") is True
        loaded = cm.load_checkpoint("B-001", "SYN-S001", "generation")
        assert loaded["patient_id"] == "PT-001"

        cm.clear_batch("B-001")
        assert cm.has_checkpoint("B-001", "SYN-S001", "generation") is False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
