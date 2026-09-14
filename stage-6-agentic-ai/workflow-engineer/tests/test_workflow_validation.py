"""Unit tests for WorkflowValidator and template registry discovery."""
from planner.workflow_planner import WorkflowPlanner
from engine.workflow_validator import validate_workflow
from workflows.registry import WorkflowRegistry
from schemas.task import Task, TaskType
from schemas.workflow import Workflow


def test_all_five_templates_valid(registry):
    templates = registry.list_templates()
    assert len(templates) >= 5

    for t_info in templates:
        wf = registry.get_template(t_info["workflow_id"])
        val = validate_workflow(wf)
        assert val.valid is True, f"Template {wf.workflow_id} failed validation: {val.errors}"
        assert val.task_count > 0
        assert len(val.execution_order) == val.task_count


def test_revenue_decline_workflow_e2e(planner):
    wf = planner.plan("Analyze this sales dataset and determine why revenue decreased during the last quarter.")
    val = validate_workflow(wf)
    assert val.valid is True
    assert val.task_count == 13
    assert val.entry_task == "T001"
    assert "T013" in val.terminal_tasks


def test_validator_detects_duplicate_tasks(sample_linear_tasks):
    # Duplicate T002
    tasks = list(sample_linear_tasks) + [sample_linear_tasks[1]]
    wf = Workflow(
        workflow_id="WF-DUP",
        name="Duplicate Task WF",
        goal="Testing duplicate IDs",
        entry_task="T001",
        terminal_tasks=["T003"],
        tasks=tasks,
    )
    val = validate_workflow(wf)
    assert val.valid is False
    assert any("Duplicate task IDs" in err for err in val.errors)


def test_validator_detects_invalid_entry_task(sample_linear_tasks):
    wf = Workflow(
        workflow_id="WF-BAD-ENTRY",
        name="Bad Entry",
        goal="Testing bad entry",
        entry_task="T999_NON_EXISTENT",
        terminal_tasks=["T003"],
        tasks=sample_linear_tasks,
    )
    val = validate_workflow(wf)
    assert val.valid is False
    assert any("Entry task" in err for err in val.errors)


def test_validator_detects_unreachable_task(sample_linear_tasks):
    orphan = Task(
        task_id="T099",
        name="Orphan",
        description="Isolated",
        task_type=TaskType.REPORTING,
        tool="report",
        dependencies=[],
    )
    wf = Workflow(
        workflow_id="WF-UNREACHABLE",
        name="Unreachable WF",
        goal="Testing unreachable",
        entry_task="T001",
        terminal_tasks=["T003"],
        tasks=sample_linear_tasks + [orphan],
    )
    val = validate_workflow(wf)
    assert val.valid is False
    assert any("unreachable" in err.lower() for err in val.errors)
