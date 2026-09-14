"""Agent execution state and lifecycle models."""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    """Lifecycle statuses for workflow agent execution runs."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    WAITING_FOR_HUMAN = "waiting_for_human"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"


class AgentMetrics(BaseModel):
    """Observability metrics for workflow run execution."""
    total_tasks: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    retries: int = 0
    escalations: int = 0
    human_overrides: int = 0
    duration_seconds: float = 0.0

    @property
    def completed_tasks(self) -> int:
        return self.completed


class AgentState(BaseModel):
    """Complete, serializable execution state of an agent run."""
    run_id: str = Field(..., description="Unique run identifier e.g. RUN-001")
    workflow_id: str = Field(..., description="Target workflow ID e.g. WF-REVENUE-001")
    current_task_id: Optional[str] = Field(None, description="Active task being executed")
    completed_tasks: List[str] = Field(default_factory=list, description="IDs of completed tasks")
    failed_tasks: List[str] = Field(default_factory=list, description="IDs of failed tasks")
    skipped_tasks: List[str] = Field(default_factory=list, description="IDs of skipped tasks")
    observations: Dict[str, Any] = Field(default_factory=dict, description="task_id -> structured observation output")
    artifacts: Dict[str, Any] = Field(default_factory=dict, description="artifact_id -> artifact metadata")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Runtime context variables (e.g. missing_rate)")
    pending_approvals: List[Dict[str, Any]] = Field(default_factory=list, description="Pending human review requests")
    escalations: List[Dict[str, Any]] = Field(default_factory=list, description="Triggered escalation events")
    status: RunStatus = Field(RunStatus.IDLE, description="High-level execution status")
    metrics: AgentMetrics = Field(default_factory=AgentMetrics, description="Execution metrics")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata and config")

    def transition_status(self, new_status: RunStatus) -> None:
        valid_transitions = {
            RunStatus.IDLE: [RunStatus.RUNNING, RunStatus.INITIALIZING, RunStatus.WAITING_FOR_HUMAN, RunStatus.FAILED],
            RunStatus.INITIALIZING: [RunStatus.RUNNING, RunStatus.WAITING_FOR_HUMAN, RunStatus.FAILED],
            RunStatus.RUNNING: [RunStatus.WAITING_FOR_HUMAN, RunStatus.PAUSED, RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.ESCALATED],
            RunStatus.WAITING_FOR_HUMAN: [RunStatus.RUNNING, RunStatus.COMPLETED, RunStatus.FAILED],
            RunStatus.PAUSED: [RunStatus.RUNNING, RunStatus.FAILED],
            RunStatus.ESCALATED: [RunStatus.WAITING_FOR_HUMAN, RunStatus.RUNNING, RunStatus.FAILED],
            RunStatus.COMPLETED: [],
            RunStatus.FAILED: [],
        }
        allowed = valid_transitions.get(self.status, [])
        if new_status not in allowed:
            raise ValueError(f"Invalid state transition from '{self.status}' to '{new_status}'.")
        self.status = new_status

    def update_variable(self, key: str, value: Any) -> None:
        self.variables[key] = value

    def record_observation(self, task_id: str, observation: Any) -> None:
        self.observations[task_id] = observation

    def record_artifact(self, artifact_id: str, artifact_meta: Dict[str, Any]) -> None:
        self.artifacts[artifact_id] = artifact_meta

    def mark_task_completed(self, task_id: str) -> None:
        if task_id not in self.completed_tasks:
            self.completed_tasks.append(task_id)
        self.metrics.completed = len(self.completed_tasks)

    def mark_task_failed(self, task_id: str) -> None:
        if task_id not in self.failed_tasks:
            self.failed_tasks.append(task_id)
        self.metrics.failed = len(self.failed_tasks)

    def record_task_failure(self, task_id: str, error: str = "") -> None:
        self.mark_task_failed(task_id)
        self.metadata[f"error_{task_id}"] = error

    def mark_task_skipped(self, task_id: str) -> None:
        if task_id not in self.skipped_tasks:
            self.skipped_tasks.append(task_id)
        self.metrics.skipped = len(self.skipped_tasks)
