"""Pytest configuration and test fixtures for Agent Engineer."""
import os
import sys
import types
from pathlib import Path
import pytest

TESTS_DIR = Path(__file__).resolve().parent
AGENT_DIR = TESTS_DIR.parent
STAGE6_DIR = AGENT_DIR.parent
WF_DIR = STAGE6_DIR / "workflow-engineer"
KE_DIR = STAGE6_DIR / "knowledge-engineer"

sys.path.insert(0, str(KE_DIR))
sys.path.insert(0, str(WF_DIR))
sys.path.insert(0, str(AGENT_DIR))

import schemas
for d in [AGENT_DIR, WF_DIR, KE_DIR]:
    sp = str(d / "schemas")
    if (d / "schemas").exists() and sp not in schemas.__path__:
        schemas.__path__.append(sp)

import tools
for d in [AGENT_DIR, KE_DIR]:
    tp = str(d / "tools")
    if (d / "tools").exists() and tp not in tools.__path__:
        tools.__path__.append(tp)

if "workflow_engineer" not in sys.modules and WF_DIR.exists():
    wf_pkg = types.ModuleType("workflow_engineer")
    wf_pkg.__path__ = [str(WF_DIR)]
    sys.modules["workflow_engineer"] = wf_pkg

if "knowledge_engineer" not in sys.modules and KE_DIR.exists():
    ke_pkg = types.ModuleType("knowledge_engineer")
    ke_pkg.__path__ = [str(KE_DIR)]
    sys.modules["knowledge_engineer"] = ke_pkg

from workflow_engineer.schemas.workflow import Workflow
from workflow_engineer.schemas.task import Task, TaskType, TaskStatus, FailurePolicy
from workflow_engineer.schemas.condition import BranchCondition, ComparisonOperator, BranchAction
from workflow_engineer.schemas.escalation import EscalationRule, EscalationTrigger, EscalationAction
from tools.registry import ToolRegistry, get_default_registry
from llm.mock import MockLLMProvider
from trace.repository import TraceRepository
from trace.recorder import TraceRecorder


@pytest.fixture
def mock_llm():
    return MockLLMProvider()


@pytest.fixture
def default_registry():
    return get_default_registry()


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_agent_traces.db"
    return str(db_file)


@pytest.fixture
def trace_repo(temp_db):
    return TraceRepository(db_path=temp_db)


@pytest.fixture
def trace_recorder(trace_repo):
    return TraceRecorder(repository=trace_repo)


@pytest.fixture
def sample_workflow():
    return Workflow(
        workflow_id="WF-TEST-001",
        name="Test Analytical Pipeline",
        version="1.0",
        goal="Analyze dataset quality and conduct exploratory analysis",
        entry_task="T001",
        terminal_tasks=["T003"],
        tasks=[
            Task(
                task_id="T001",
                name="Load Data",
                description="Load sales records from file",
                task_type=TaskType.INGESTION,
                tool="load_dataset",
                dependencies=[],
                expected_outputs=["raw_dataset"],
            ),
            Task(
                task_id="T002",
                name="Profile Data",
                description="Profile dataset statistics",
                task_type=TaskType.PROFILING,
                tool="profile_dataset",
                dependencies=["T001"],
                expected_outputs=["profile_summary"],
            ),
            Task(
                task_id="T003",
                name="EDA Analysis",
                description="Perform bivariate revenue analysis",
                task_type=TaskType.EDA,
                tool="eda_analysis",
                dependencies=["T002"],
                expected_outputs=["eda_insights"],
            ),
        ],
    )


@pytest.fixture
def branching_workflow():
    return Workflow(
        workflow_id="WF-BRANCH-001",
        name="Branching Pipeline",
        version="1.0",
        goal="Test dynamic branching logic on data quality",
        entry_task="T001",
        terminal_tasks=["T005"],
        tasks=[
            Task(
                task_id="T001",
                name="Load Data",
                description="Load sales records",
                task_type=TaskType.INGESTION,
                tool="load_dataset",
                dependencies=[],
            ),
            Task(
                task_id="T002",
                name="Validate Data",
                description="Assess data quality score",
                task_type=TaskType.PROFILING,
                tool="validate_data_quality",
                dependencies=["T001"],
                conditions=[
                    BranchCondition(
                        condition_id="COND-01",
                        metric_name="quality_score",
                        operator=ComparisonOperator.GT,
                        threshold=0.80,
                        action=BranchAction.CONTINUE_ANALYSIS,
                        target_task_id="T005",
                        description="Skip cleaning if quality > 0.80",
                    )
                ],
            ),
            Task(
                task_id="T004",
                name="Clean Data",
                description="Impute nulls and remove outliers",
                task_type=TaskType.PROFILING,
                tool="clean_dataset",
                dependencies=["T002"],
            ),
            Task(
                task_id="T005",
                name="Final Report",
                description="Synthesize report",
                task_type=TaskType.REPORTING,
                tool="generate_report",
                dependencies=["T002"],
            ),
        ],
    )


@pytest.fixture
def escalation_workflow():
    return Workflow(
        workflow_id="WF-ESCALATE-001",
        name="Escalation Pipeline",
        version="1.0",
        goal="Test automated human review triggers",
        entry_task="T001",
        terminal_tasks=["T002"],
        tasks=[
            Task(
                task_id="T001",
                name="Check Quality",
                description="Validate data quality",
                task_type=TaskType.PROFILING,
                tool="validate_data_quality",
                dependencies=[],
            ),
            Task(
                task_id="T002",
                name="Train Predictive Model",
                description="Train regression model",
                task_type=TaskType.MODEL_TRAINING,
                tool="train_model",
                dependencies=["T001"],
            ),
        ],
        escalation_rules=[
            EscalationRule(
                rule_id="ESC-LOW-QUAL",
                name="Low Data Quality Check",
                trigger=EscalationTrigger.POOR_DATA_QUALITY,
                condition_metric="quality_score",
                operator=ComparisonOperator.LT,
                threshold=0.70,
                action=EscalationAction.HUMAN_REVIEW,
                message="Data quality too low for model training.",
            )
        ],
    )
