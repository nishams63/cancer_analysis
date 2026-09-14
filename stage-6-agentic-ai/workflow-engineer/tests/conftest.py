"""Pytest fixtures and test graph generators."""
import sys
from pathlib import Path
import pytest

# Ensure package root is in sys.path
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from schemas.task import Task, TaskType, TaskStatus, TaskPriority, FailurePolicy
from schemas.workflow import Workflow
from planner.workflow_planner import WorkflowPlanner
from workflows.registry import WorkflowRegistry


@pytest.fixture
def sample_linear_tasks():
    return [
        Task(
            task_id="T001",
            name="Load Data",
            description="Ingest raw data",
            task_type=TaskType.INGESTION,
            tool="load_dataset",
            dependencies=[],
            expected_outputs=["raw_df"],
        ),
        Task(
            task_id="T002",
            name="Profile Data",
            description="Profile columns",
            task_type=TaskType.PROFILING,
            tool="profile_dataset",
            dependencies=["T001"],
            required_inputs=["raw_df"],
            expected_outputs=["profile"],
        ),
        Task(
            task_id="T003",
            name="Generate Report",
            description="Report findings",
            task_type=TaskType.REPORTING,
            tool="generate_report",
            dependencies=["T002"],
            required_inputs=["profile"],
            expected_outputs=["report"],
        ),
    ]


@pytest.fixture
def sample_cyclic_tasks():
    return [
        Task(
            task_id="T001",
            name="Task A",
            description="First task",
            task_type=TaskType.INGESTION,
            tool="load_dataset",
            dependencies=["T003"],  # Cycle: T001 -> T002 -> T003 -> T001
        ),
        Task(
            task_id="T002",
            name="Task B",
            description="Second task",
            task_type=TaskType.PROFILING,
            tool="profile_dataset",
            dependencies=["T001"],
        ),
        Task(
            task_id="T003",
            name="Task C",
            description="Third task",
            task_type=TaskType.EDA,
            tool="eda_analysis",
            dependencies=["T002"],
        ),
    ]


@pytest.fixture
def planner():
    return WorkflowPlanner()


@pytest.fixture
def registry():
    return WorkflowRegistry()
