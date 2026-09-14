"""Run lifecycle state definitions and transition validations."""
from __future__ import annotations
from enum import Enum
from typing import Set


class RunLifecycleState(str, Enum):
    CREATED = "CREATED"
    PLANNING = "PLANNING"
    READY = "READY"
    RUNNING = "RUNNING"
    WAITING_FOR_HUMAN = "WAITING_FOR_HUMAN"
    EVALUATING = "EVALUATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# Valid state transitions
VALID_TRANSITIONS: dict[RunLifecycleState, Set[RunLifecycleState]] = {
    RunLifecycleState.CREATED: {RunLifecycleState.PLANNING, RunLifecycleState.FAILED, RunLifecycleState.CANCELLED},
    RunLifecycleState.PLANNING: {RunLifecycleState.READY, RunLifecycleState.FAILED, RunLifecycleState.CANCELLED},
    RunLifecycleState.READY: {RunLifecycleState.RUNNING, RunLifecycleState.FAILED, RunLifecycleState.CANCELLED},
    RunLifecycleState.RUNNING: {
        RunLifecycleState.WAITING_FOR_HUMAN,
        RunLifecycleState.EVALUATING,
        RunLifecycleState.COMPLETED,
        RunLifecycleState.FAILED,
        RunLifecycleState.CANCELLED,
    },
    RunLifecycleState.WAITING_FOR_HUMAN: {
        RunLifecycleState.RUNNING,
        RunLifecycleState.COMPLETED,
        RunLifecycleState.FAILED,
        RunLifecycleState.CANCELLED,
    },
    RunLifecycleState.EVALUATING: {
        RunLifecycleState.COMPLETED,
        RunLifecycleState.FAILED,
    },
    RunLifecycleState.COMPLETED: set(),
    RunLifecycleState.FAILED: set(),
    RunLifecycleState.CANCELLED: set(),
}


def can_transition(current: RunLifecycleState, target: RunLifecycleState) -> bool:
    """Validate whether current lifecycle state can legally transition to target."""
    return target in VALID_TRANSITIONS.get(current, set())
