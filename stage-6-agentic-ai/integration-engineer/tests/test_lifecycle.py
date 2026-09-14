"""Tests for run lifecycle state machine."""
from orchestration.lifecycle import RunLifecycleState, can_transition


def test_run_lifecycle_valid_transitions():
    assert can_transition(RunLifecycleState.CREATED, RunLifecycleState.PLANNING)
    assert can_transition(RunLifecycleState.PLANNING, RunLifecycleState.READY)
    assert can_transition(RunLifecycleState.READY, RunLifecycleState.RUNNING)
    assert can_transition(RunLifecycleState.RUNNING, RunLifecycleState.WAITING_FOR_HUMAN)
    assert can_transition(RunLifecycleState.WAITING_FOR_HUMAN, RunLifecycleState.RUNNING)
    assert can_transition(RunLifecycleState.RUNNING, RunLifecycleState.EVALUATING)
    assert can_transition(RunLifecycleState.EVALUATING, RunLifecycleState.COMPLETED)


def test_run_lifecycle_invalid_transitions():
    assert not can_transition(RunLifecycleState.COMPLETED, RunLifecycleState.RUNNING)
    assert not can_transition(RunLifecycleState.FAILED, RunLifecycleState.PLANNING)
    assert not can_transition(RunLifecycleState.CREATED, RunLifecycleState.COMPLETED)
