"""Asynchronous run coordinator managing background execution threads."""
from typing import Optional, Dict, Any
import threading
from .pipeline import AADAIntegrationPipeline
from .session_manager import AADARunSession, get_session_manager


class AADACoordinator:
    """Manages asynchronous non-blocking submission and execution of AADA pipelines."""

    def __init__(self, pipeline: Optional[AADAIntegrationPipeline] = None):
        self.pipeline = pipeline or AADAIntegrationPipeline()
        self.session_manager = get_session_manager()

    def start_run_async(self, goal: str, run_id: Optional[str] = None) -> str:
        """Kick off a pipeline run in a background worker thread and return run_id immediately."""
        import uuid
        actual_run_id = run_id or f"RUN-{uuid.uuid4().hex[:8].upper()}"

        # Initialize session record immediately so queries don't 404
        self.session_manager.create_session(run_id=actual_run_id, goal=goal)

        thread = threading.Thread(
            target=self.pipeline.run_pipeline,
            kwargs={"goal": goal, "custom_run_id": actual_run_id},
            daemon=True,
        )
        thread.start()
        return actual_run_id


_GLOBAL_COORDINATOR = AADACoordinator()

def get_global_coordinator() -> AADACoordinator:
    return _GLOBAL_COORDINATOR
