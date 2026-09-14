"""Unit tests for Pydantic Workflow and Task schemas."""
import pytest
from pydantic import ValidationError
from schemas.task import (
    Task,
    TaskType,
    TaskStatus,
    FailurePolicy,
    RetryPolicy,
    TaskPriority,
    KnowledgeRetrievalSpec,
)
from schemas.workflow import Workflow, WorkflowInput, WorkflowOutput
from schemas.condition import BranchCondition, ComparisonOperator, BranchAction
from schemas.escalation import EscalationRule, EscalationTrigger, EscalationAction


def test_valid_task_creation():
    task = Task(
        task_id="T001",
        name="Ingest Dataset",
        description="Load CSV into DataFrame",
        task_type=TaskType.INGESTION,
        tool="load_dataset",
        dependencies=[],
        expected_outputs=["dataframe"],
    )
    assert task.task_id == "T001"
    assert task.task_type == TaskType.INGESTION
    assert task.status == TaskStatus.PENDING
    assert task.failure_policy == FailurePolicy.RETRY
    assert task.retry_policy.max_retries == 3


def test_invalid_task_type_rejected():
    with pytest.raises(ValidationError):
        Task(
            task_id="T001",
            name="Bad Task",
            description="Invalid task type",
            task_type="invalid_future_type",  # Not in TaskType enum
            tool="some_tool",
        )


def test_knowledge_retrieval_spec():
    spec = KnowledgeRetrievalSpec(
        query="Outlier detection IQR",
        category="anomaly_detection",
        top_k=3,
    )
    assert spec.top_k == 3
    assert spec.category == "anomaly_detection"


def test_workflow_serialization(sample_linear_tasks):
    wf = Workflow(
        workflow_id="WF-TEST-001",
        name="Test Linear Workflow",
        goal="Test pipeline execution",
        entry_task="T001",
        terminal_tasks=["T003"],
        tasks=sample_linear_tasks,
    )
    json_str = wf.to_json()
    assert "WF-TEST-001" in json_str

    wf_loaded = Workflow.from_json(json_str)
    assert wf_loaded.workflow_id == "WF-TEST-001"
    assert len(wf_loaded.tasks) == 3
    assert wf_loaded.get_task("T002").name == "Profile Data"


def test_workflow_graph_helpers(sample_linear_tasks):
    wf = Workflow(
        workflow_id="WF-TEST-002",
        name="Test Helpers",
        goal="Test helper functions",
        entry_task="T001",
        terminal_tasks=["T003"],
        tasks=sample_linear_tasks,
    )
    succ = wf.get_successors("T001")
    assert len(succ) == 1
    assert succ[0].task_id == "T002"

    pred = wf.get_predecessors("T003")
    assert len(pred) == 1
    assert pred[0].task_id == "T002"
