"""Tests for AgentState, AgentMetrics, and AgentStateManager."""
import pytest
from schemas.agent import AgentState, RunStatus, AgentMetrics
from agent.state import AgentStateManager
from trace.recorder import TraceRecorder


def test_state_initialization():
    state = AgentState(run_id="RUN-001", workflow_id="WF-001")
    assert state.status == RunStatus.IDLE
    assert state.metrics.total_tasks == 0
    assert state.current_task_id is None
    assert len(state.completed_tasks) == 0


def test_state_transitions():
    state = AgentState(run_id="RUN-002", workflow_id="WF-001")
    state.transition_status(RunStatus.RUNNING)
    assert state.status == RunStatus.RUNNING

    state.transition_status(RunStatus.WAITING_FOR_HUMAN)
    assert state.status == RunStatus.WAITING_FOR_HUMAN

    state.transition_status(RunStatus.COMPLETED)
    assert state.status == RunStatus.COMPLETED


def test_invalid_state_transition():
    state = AgentState(run_id="RUN-003", workflow_id="WF-001", status=RunStatus.COMPLETED)
    with pytest.raises(ValueError):
        state.transition_status(RunStatus.RUNNING)


def test_state_variable_tracking():
    state = AgentState(run_id="RUN-004", workflow_id="WF-001")
    state.update_variable("quality_score", 0.92)
    state.record_observation("T001", {"rows": 5000, "cols": 12})
    state.mark_task_completed("T001")

    assert state.variables["quality_score"] == 0.92
    assert "T001" in state.completed_tasks
    assert state.metrics.completed_tasks == 1
    assert state.observations["T001"]["rows"] == 5000


def test_state_manager_lifecycle(trace_recorder):
    manager = AgentStateManager.create_initial(
        run_id="RUN-005",
        workflow_id="WF-001",
        total_tasks=3,
        recorder=trace_recorder,
    )
    assert manager.state.status == RunStatus.RUNNING
    assert manager.state.metrics.total_tasks == 3

    manager.finish_run(RunStatus.COMPLETED)
    assert manager.state.status == RunStatus.COMPLETED
    assert "completed_at" in manager.state.metadata
