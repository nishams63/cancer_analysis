"""Branch condition evaluation against agent state."""
from typing import Tuple, Optional
from schemas.agent import AgentState
from schemas.condition import BranchAction
from workflow_engineer.schemas.task import Task


def evaluate_task_branch(
    task: Task, state: AgentState
) -> Tuple[Optional[BranchAction], Optional[str]]:
    """Evaluate task branch conditions against runtime variables."""
    context = dict(state.variables)
    # Also inject observation metrics from current task if available
    if task.task_id in state.observations:
        obs = state.observations[task.task_id]
        if isinstance(obs, dict):
            context.update(obs)

    for cond in task.conditions:
        if cond.evaluate(context):
            return cond.action, cond.target_task_id

    return None, None
