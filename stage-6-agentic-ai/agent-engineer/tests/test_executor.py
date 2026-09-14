"""Tests for authoritative AgentExecutor orchestrating full DAG pipelines."""
from agent.executor import AgentExecutor
from schemas.agent import RunStatus
from workflow_engineer.schemas.workflow import Workflow
from workflow_engineer.schemas.task import Task, TaskType, FailurePolicy
from workflow_engineer.schemas.condition import ComparisonOperator
from workflow_engineer.schemas.escalation import EscalationRule, EscalationTrigger, EscalationAction


def test_execute_sample_workflow(sample_workflow, default_registry, mock_llm, trace_repo):
    executor = AgentExecutor(
        tool_registry=default_registry,
        llm_provider=mock_llm,
        trace_repository=trace_repo,
    )
    result = executor.run(sample_workflow)

    assert result.status == "completed"
    assert len(result.completed_tasks) == 3
    assert result.confidence > 0.70
    assert result.trace_id is not None
    assert result.metrics["completed"] == 3


def test_conditional_branching_skip(branching_workflow, default_registry, mock_llm, trace_repo):
    executor = AgentExecutor(
        tool_registry=default_registry,
        llm_provider=mock_llm,
        trace_repository=trace_repo,
    )
    result = executor.run(branching_workflow)

    assert result.status == "completed"
    assert "T004" not in result.completed_tasks
    assert "T005" in result.completed_tasks


def test_human_pause_and_resume(default_registry, mock_llm, trace_repo):
    escalation_wf = Workflow(
        workflow_id="WF-ESC-PAUSE",
        name="Escalation Pause Pipeline",
        goal="Test pause on low quality",
        entry_task="T001",
        terminal_tasks=["T002"],
        tasks=[
            Task(
                task_id="T001",
                name="Validate Data",
                description="Check quality score",
                task_type=TaskType.PROFILING,
                tool="validate_data_quality",
                dependencies=[],
            ),
            Task(
                task_id="T002",
                name="Model",
                description="Model task execution",
                task_type=TaskType.MACHINE_LEARNING,
                tool="train_model",
                dependencies=["T001"],
            ),
        ],
        escalation_rules=[
            EscalationRule(
                rule_id="ESC-FORCE-PAUSE",
                name="Force Pause",
                trigger=EscalationTrigger.POOR_DATA_QUALITY,
                condition_metric="quality_score",
                operator=ComparisonOperator.GT,
                threshold=0.50,
                action=EscalationAction.HUMAN_REVIEW,
                message="Force pause for testing human in the loop",
            )
        ],
    )
    executor = AgentExecutor(
        tool_registry=default_registry,
        llm_provider=mock_llm,
        trace_repository=trace_repo,
    )
    result = executor.run(escalation_wf)
    assert result.status == RunStatus.WAITING_FOR_HUMAN.value
    assert result.escalated is True

    resumed = executor.resume_after_approval(
        run_id=result.run_id,
        workflow=escalation_wf,
        decision="approve",
        reason="Analyst verified model training bounds",
    )
    assert resumed.status == RunStatus.COMPLETED.value
    assert "T002" in resumed.completed_tasks


def test_failed_task_and_skip_dependents(default_registry, mock_llm, trace_repo):
    failing_wf = Workflow(
        workflow_id="WF-FAIL-01",
        name="Failing Pipeline",
        goal="Verify skip dependents on failure",
        entry_task="T001",
        terminal_tasks=["T003"],
        tasks=[
            Task(
                task_id="T001",
                name="Fail Task",
                description="Simulated fail execution",
                task_type=TaskType.INGESTION,
                tool="load_dataset",
                dependencies=[],
                failure_policy=FailurePolicy.SKIP,
            ),
            Task(
                task_id="T002",
                name="Downstream Task",
                description="Dependent task execution",
                task_type=TaskType.PROFILING,
                tool="profile_dataset",
                dependencies=["T001"],
            ),
            Task(
                task_id="T003",
                name="Terminal Task",
                description="Leaf task execution",
                task_type=TaskType.REPORTING,
                tool="generate_report",
                dependencies=["T002"],
            ),
        ],
    )
    def broken_loader(inputs):
        return {"status": "error", "error": "Disk read failure"}
    default_registry.register("load_dataset", broken_loader)

    executor = AgentExecutor(
        tool_registry=default_registry,
        llm_provider=mock_llm,
        trace_repository=trace_repo,
    )
    result = executor.run(failing_wf)
    assert "T001" not in result.completed_tasks
    assert "T002" not in result.completed_tasks
    assert "T003" not in result.completed_tasks
