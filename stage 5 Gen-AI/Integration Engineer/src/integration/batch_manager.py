from enum import Enum
from typing import Dict, Any, Optional
from ..storage.batch_store import BatchStore
from ..utils.logging import get_logger

logger = get_logger("BatchManager")

class BatchState(str, Enum):
    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    GENERATING = "GENERATING"
    RETRIEVING = "RETRIEVING"
    NARRATIVE_GENERATION = "NARRATIVE_GENERATION"
    EVALUATING = "EVALUATING"
    RUNNING_STAGE1 = "RUNNING_STAGE1"
    RUNNING_STAGE2 = "RUNNING_STAGE2"
    RUNNING_STAGE3 = "RUNNING_STAGE3"
    RUNNING_STAGE4 = "RUNNING_STAGE4"
    ANALYZING_FAILURES = "ANALYZING_FAILURES"
    RANKING = "RANKING"
    COMPLETED = "COMPLETED"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"
    FAILED = "FAILED"
    FAILED_DEPENDENCY_CHECK = "FAILED_DEPENDENCY_CHECK"

class BatchManager:
    def __init__(self, batch_store: BatchStore):
        self.batch_store = batch_store

    def start_batch(self, batch_id: str, seed: int, requested_count: int, config_versions: Dict[str, Any], system_manifest: Dict[str, Any]) -> None:
        self.batch_store.create_batch(batch_id, seed, requested_count, config_versions, system_manifest)
        logger.info(f"Initialized batch {batch_id} (seed={seed}, count={requested_count})")

    def transition(self, batch_id: str, state: BatchState, actual_count: Optional[int] = None) -> None:
        logger.info(f"Batch {batch_id} transitioning to: {state.value}")
        self.batch_store.update_status(batch_id, state.value, actual_count)
