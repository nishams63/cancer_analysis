"""Workflow Graph and Schema Validation Engine."""
from __future__ import annotations
from typing import List, Dict, Set, Optional, Any
from pydantic import BaseModel, Field
from schemas.workflow import Workflow
from schemas.task import Task, TaskType
from planner.dependency_resolver import DependencyResolver


class ValidationResult(BaseModel):
    """Structured outcome of workflow graph and schema inspection."""
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    task_count: int = 0
    entry_task: Optional[str] = None
    terminal_tasks: List[str] = Field(default_factory=list)
    execution_order: List[str] = Field(default_factory=list)


def validate_workflow(workflow: Workflow) -> ValidationResult:
    """Comprehensively inspect a Workflow for structural and semantic validity."""
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Unique Workflow ID
    if not workflow.workflow_id or not workflow.workflow_id.strip():
        errors.append("Workflow ID is missing or empty.")

    # 2. Task Count & Unique Task IDs
    if not workflow.tasks:
        errors.append("Workflow contains zero tasks.")
        return ValidationResult(valid=False, errors=errors, task_count=0)

    task_ids: Set[str] = set()
    duplicate_ids: Set[str] = set()
    task_map: Dict[str, Task] = {}

    for t in workflow.tasks:
        if t.task_id in task_ids:
            duplicate_ids.add(t.task_id)
        task_ids.add(t.task_id)
        task_map[t.task_id] = t

    if duplicate_ids:
        errors.append(f"Duplicate task IDs found: {sorted(list(duplicate_ids))}")

    # 3. Valid Entry Task
    if not workflow.entry_task or workflow.entry_task not in task_ids:
        errors.append(f"Entry task '{workflow.entry_task}' does not exist in workflow tasks.")
    else:
        entry_t = task_map[workflow.entry_task]
        if entry_t.dependencies:
            warnings.append(f"Entry task '{workflow.entry_task}' has parent dependencies: {entry_t.dependencies}")

    # 4. Valid Terminal Tasks
    if not workflow.terminal_tasks:
        errors.append("Workflow has no terminal tasks specified.")
    else:
        for term_id in workflow.terminal_tasks:
            if term_id not in task_ids:
                errors.append(f"Terminal task '{term_id}' does not exist in workflow tasks.")

    # 5. Validate Task Dependencies & Missing Parent References
    for t in workflow.tasks:
        for dep in t.dependencies:
            if dep not in task_ids:
                errors.append(f"Task '{t.task_id}' references non-existent parent dependency '{dep}'.")
            if dep == t.task_id:
                errors.append(f"Task '{t.task_id}' cannot depend on itself.")

    # 6. Cycle Detection
    has_cycle, cycle_msg = DependencyResolver.detect_cycles(workflow.tasks)
    if has_cycle:
        errors.append(cycle_msg or "Circular dependency detected.")

    # 7. Reachability from Entry Task
    if not has_cycle and workflow.entry_task in task_ids:
        unreachable = DependencyResolver.find_unreachable_tasks(workflow.tasks, workflow.entry_task)
        if unreachable:
            errors.append(f"Tasks unreachable from entry task '{workflow.entry_task}': {unreachable}")

    # 8. Topological Sort Order
    exec_order: List[str] = []
    if not errors:
        try:
            exec_order = DependencyResolver.topological_sort(workflow.tasks)
        except Exception as e:
            errors.append(f"Failed to resolve execution order: {e}")

    # 9. Individual Task Field Invariants
    for t in workflow.tasks:
        if not t.tool or not t.tool.strip():
            errors.append(f"Task '{t.task_id}' must specify a valid tool name.")

        # Validate Branch Conditions
        for cond in t.conditions:
            if cond.target_task_id and cond.target_task_id not in task_ids:
                errors.append(
                    f"Task '{t.task_id}' condition '{cond.condition_id}' targets non-existent task '{cond.target_task_id}'."
                )

        # Validate Knowledge Retrieval Spec
        if t.knowledge_retrieval:
            if not t.knowledge_retrieval.query or not t.knowledge_retrieval.query.strip():
                errors.append(f"Task '{t.task_id}' specifies empty knowledge retrieval query.")

    # 10. Escalation Rules Invariants
    for rule in workflow.escalation_rules:
        if not rule.condition_metric or not rule.condition_metric.strip():
            errors.append(f"Escalation rule '{rule.rule_id}' has missing condition_metric.")

    is_valid = len(errors) == 0
    return ValidationResult(
        valid=is_valid,
        errors=errors,
        warnings=warnings,
        task_count=len(workflow.tasks),
        entry_task=workflow.entry_task if workflow.entry_task in task_ids else None,
        terminal_tasks=workflow.terminal_tasks,
        execution_order=exec_order,
    )
