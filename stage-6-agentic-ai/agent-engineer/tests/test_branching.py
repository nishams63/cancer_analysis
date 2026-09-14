"""Tests for task branch condition evaluation."""
from schemas.agent import AgentState
from workflow_engineer.schemas.task import Task, TaskType
from workflow_engineer.schemas.condition import BranchCondition, ComparisonOperator, BranchAction
from execution.branch_executor import evaluate_task_branch


def test_branch_condition_greater_than_satisfied():
    task = Task(
        task_id="T002",
        name="Validate Quality",
        description="Check score",
        task_type=TaskType.PROFILING,
        tool="validate_data_quality",
        conditions=[
            BranchCondition(
                condition_id="COND-1",
                metric_name="quality_score",
                operator=ComparisonOperator.GT,
                threshold=0.85,
                action=BranchAction.CONTINUE_ANALYSIS,
                target_task_id="T005",
            )
        ],
    )
    state = AgentState(run_id="RUN-01", workflow_id="WF-01")
    state.update_variable("quality_score", 0.90)

    action, target = evaluate_task_branch(task, state)
    assert action == BranchAction.CONTINUE_ANALYSIS
    assert target == "T005"


def test_branch_condition_not_satisfied():
    task = Task(
        task_id="T002",
        name="Validate Quality",
        description="Check score",
        task_type=TaskType.PROFILING,
        tool="validate_data_quality",
        conditions=[
            BranchCondition(
                condition_id="COND-1",
                metric_name="quality_score",
                operator=ComparisonOperator.GT,
                threshold=0.85,
                action=BranchAction.CONTINUE_ANALYSIS,
                target_task_id="T005",
            )
        ],
    )
    state = AgentState(run_id="RUN-01", workflow_id="WF-01")
    state.update_variable("quality_score", 0.75)

    action, target = evaluate_task_branch(task, state)
    assert action is None
    assert target is None
