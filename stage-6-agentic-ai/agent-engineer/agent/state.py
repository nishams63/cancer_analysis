"""State manager handling lifecycle transitions and variable tracking."""
from __future__ import annotations
import time
from datetime import datetime
from typing import Dict, Any, Optional
from schemas.agent import AgentState, RunStatus, AgentMetrics
from trace.recorder import TraceRecorder
from trace.repository import TraceRepository


class AgentStateManager:
    """Manages and serializes AgentState throughout a run."""

    def __init__(self, state: AgentState, recorder: TraceRecorder):
        self.state = state
        self.recorder = recorder
        self._start_time = time.perf_counter()

    @classmethod
    def create_initial(
        cls, run_id: str, workflow_id: str, total_tasks: int, recorder: TraceRecorder
    ) -> AgentStateManager:
        now_iso = datetime.utcnow().isoformat() + "Z"
        metrics = AgentMetrics(total_tasks=total_tasks)
        state = AgentState(
            run_id=run_id,
            workflow_id=workflow_id,
            status=RunStatus.RUNNING,
            metrics=metrics,
            metadata={"started_at": now_iso},
        )
        return cls(state, recorder)

    def finish_run(self, status: RunStatus = RunStatus.COMPLETED) -> None:
        self.state.status = status
        now_iso = datetime.utcnow().isoformat() + "Z"
        self.state.metadata["completed_at"] = now_iso
        self.state.metrics.duration_seconds = round(time.perf_counter() - self._start_time, 3)
        self.recorder.repo.save_run(self.state)
