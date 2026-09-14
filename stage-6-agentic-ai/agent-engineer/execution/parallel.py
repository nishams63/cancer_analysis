"""Parallel task grouping and dependency barrier synchronization."""
from typing import List, Set, Optional
from workflow_engineer.schemas.workflow import Workflow
from workflow_engineer.schemas.task import Task


def get_ready_tasks(
    workflow: Workflow,
    completed_tasks: Set[str],
    skipped_tasks: Set[str],
    in_progress_tasks: Set[str],
    failed_tasks: Optional[Set[str]] = None,
) -> List[Task]:
    """Find all tasks whose prerequisite dependencies have completed or been skipped."""
    ready = []
    satisfied_set = completed_tasks | skipped_tasks
    unavailable_set = satisfied_set | in_progress_tasks | (failed_tasks or set())

    for task in workflow.tasks:
        if task.task_id in unavailable_set:
            continue
        # All dependencies must be satisfied
        if all(dep in satisfied_set for dep in task.dependencies):
            ready.append(task)

    return ready
