"""Tests for global escalation triggers (low quality, conflicting findings)."""
from schemas.agent import AgentState
from workflow_engineer.schemas.workflow import Workflow
from workflow_engineer.schemas.task import Task, TaskType
from workflow_engineer.schemas.condition import ComparisonOperator
from workflow_engineer.schemas.escalation import EscalationRule, EscalationTrigger, EscalationAction
from execution.escalation import check_escalations


def test_escalation_on_low_data_quality():
    wf = Workflow(
        workflow_id="WF-ESC-01",
        name="Esc Workflow",
        goal="Test esc pipeline",
        entry_task="T1",
        terminal_tasks=["T1"],
        tasks=[Task(task_id="T1", name="Task 1", description="Profile dataset description", task_type=TaskType.PROFILING, tool="profile_dataset")],
        escalation_rules=[
            EscalationRule(
                rule_id="ESC-LOW-QUAL",
                name="Low Quality Rule",
                trigger=EscalationTrigger.POOR_DATA_QUALITY,
                condition_metric="quality_score",
                operator=ComparisonOperator.LT,
                threshold=0.70,
                action=EscalationAction.HUMAN_REVIEW,
                message="quality score below threshold",
            )
        ],
    )
    state = AgentState(run_id="RUN-01", workflow_id="WF-ESC-01")
    state.update_variable("quality_score", 0.65)

    res = check_escalations(wf, state)
    assert res is not None
    rule, msg = res
    assert rule.rule_id == "ESC-LOW-QUAL"
    assert "Observed: 0.65" in msg


def test_escalation_on_conflicting_findings():
    wf = Workflow(
        workflow_id="WF-ESC-02",
        name="Esc Workflow",
        goal="Test esc pipeline",
        entry_task="T1",
        terminal_tasks=["T1"],
        tasks=[Task(task_id="T1", name="Task 1", description="Profile dataset description", task_type=TaskType.PROFILING, tool="profile_dataset")],
        escalation_rules=[
            EscalationRule(
                rule_id="ESC-CONFLICT",
                name="Conflict Rule",
                trigger=EscalationTrigger.SIMILAR_CAUSE_SCORES,
                condition_metric="cause_score_margin",
                operator=ComparisonOperator.LT,
                threshold=0.10,
                action=EscalationAction.HUMAN_REVIEW,
                message="ambiguous cause score margin",
            )
        ],
    )
    state = AgentState(run_id="RUN-02", workflow_id="WF-ESC-02")
    state.update_variable("cause_score_margin", 0.05)

    res = check_escalations(wf, state)
    assert res is not None
    rule, msg = res
    assert rule.rule_id == "ESC-CONFLICT"
