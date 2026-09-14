"""Pytest configuration and shared fixtures for Evaluation Engineer."""
import sys
import types
from pathlib import Path
import pytest

TESTS_DIR = Path(__file__).resolve().parent
EVAL_DIR = TESTS_DIR.parent
STAGE6_DIR = EVAL_DIR.parent

sys.path.insert(0, str(STAGE6_DIR / "knowledge-engineer"))
sys.path.insert(0, str(STAGE6_DIR / "workflow-engineer"))
sys.path.insert(0, str(STAGE6_DIR / "agent-engineer"))
sys.path.insert(0, str(EVAL_DIR))

# Module Aliases
if "workflow_engineer" not in sys.modules:
    wf_pkg = types.ModuleType("workflow_engineer")
    wf_pkg.__path__ = [str(STAGE6_DIR / "workflow-engineer")]
    sys.modules["workflow_engineer"] = wf_pkg

if "knowledge_engineer" not in sys.modules:
    ke_pkg = types.ModuleType("knowledge_engineer")
    ke_pkg.__path__ = [str(STAGE6_DIR / "knowledge-engineer")]
    sys.modules["knowledge_engineer"] = ke_pkg

if "agent_engineer" not in sys.modules:
    ae_pkg = types.ModuleType("agent_engineer")
    ae_pkg.__path__ = [str(STAGE6_DIR / "agent-engineer")]
    sys.modules["agent_engineer"] = ae_pkg

import schemas
for d in [EVAL_DIR, STAGE6_DIR / "agent-engineer", STAGE6_DIR / "workflow-engineer"]:
    sp = str(d / "schemas")
    if (d / "schemas").exists() and sp not in schemas.__path__:
        schemas.__path__.append(sp)

from schemas.result import AgentResult
from schemas.trace import TraceEvent, EventType
from schemas.scenario import EvaluationScenario, ScenarioCategory, ScenarioDifficulty
from schemas.ground_truth import GroundTruth, ExpectedFact, AcceptableCause
from experiments.runner import ExperimentRunner


@pytest.fixture
def runner():
    return ExperimentRunner()


@pytest.fixture
def sample_scenario():
    return EvaluationScenario(
        scenario_id="SC-REV-001",
        name="Revenue Decline Analysis",
        description="Investigate revenue drop in Q3",
        category=ScenarioCategory.SALES,
        dataset="data/sales_q3.csv",
        user_goal="Determine why revenue decreased",
        expected_workflow="WF-REVENUE-001",
        expected_tasks=["T001", "T002", "T003", "T005"],
        expected_tools=["load_dataset", "profile_dataset", "retrieve_knowledge", "eda_analysis"],
        expected_knowledge=["revenue drop analysis methods"],
        expected_branches=[{"condition_id": "COND-01", "target": "T005"}],
        expected_escalations=[],
        difficulty=ScenarioDifficulty.MEDIUM,
    )


@pytest.fixture
def sample_ground_truth():
    return GroundTruth(
        ground_truth_id="GT-REV-001",
        scenario_id="SC-REV-001",
        expected_facts=[
            ExpectedFact(fact_id="F-01", description="Revenue decline rate", metric_name="decline_rate", expected_value=0.18, tolerance_pct=0.05, is_critical=True),
            ExpectedFact(fact_id="F-02", description="Decline magnitude", metric_name="decline_magnitude", expected_value=180000.0, tolerance_pct=0.05, is_critical=True)
        ],
        acceptable_causes=[
            AcceptableCause(cause="Price Increase in Enterprise Tier", min_score=0.70, max_rank=1)
        ],
        forbidden_claims=["Revenue increased during Q2 and Q3"],
        expected_recommendation_themes=["pricing", "tier"],
    )


@pytest.fixture
def sample_agent_result():
    return AgentResult(
        run_id="RUN-TEST-001",
        workflow_id="WF-REVENUE-001",
        status="completed",
        objective="Determine why revenue decreased",
        findings=[
            {"cause": "Price Increase in Enterprise Tier", "score": 0.86, "details": "decline_rate 0.18"}
        ],
        evidence=[
            {"decline_rate": 0.18, "decline_magnitude": 180000.0}
        ],
        recommendations=[
            {"recommendation": "Rebalance Enterprise pricing tiers."}
        ],
        confidence=0.85,
        escalated=False,
        completed_tasks=["T001", "T002", "T003", "T005"],
        trace_id="TRACE-001",
        metrics={"duration_seconds": 0.35, "tool_calls": 3},
    )


@pytest.fixture
def sample_trace_events():
    return [
        TraceEvent(trace_id="TR-1", run_id="RUN-TEST-001", timestamp="2026-09-13T12:00:00Z", task_id="T001", event_type=EventType.TASK_STARTED),
        TraceEvent(trace_id="TR-2", run_id="RUN-TEST-001", timestamp="2026-09-13T12:00:01Z", task_id="T001", event_type=EventType.TOOL_CALL, tool_name="load_dataset"),
        TraceEvent(trace_id="TR-3", run_id="RUN-TEST-001", timestamp="2026-09-13T12:00:02Z", task_id="T002", event_type=EventType.TOOL_CALL, tool_name="profile_dataset"),
        TraceEvent(trace_id="TR-4", run_id="RUN-TEST-001", timestamp="2026-09-13T12:00:03Z", task_id="T003", event_type=EventType.BRANCH_DECISION, next_task_id="T005"),
        TraceEvent(trace_id="TR-5", run_id="RUN-TEST-001", timestamp="2026-09-13T12:00:04Z", task_id="T005", event_type=EventType.TOOL_CALL, tool_name="retrieve_knowledge", tool_input={"query": "revenue drop analysis methods"}),
        TraceEvent(trace_id="TR-6", run_id="RUN-TEST-001", timestamp="2026-09-13T12:00:05Z", task_id="T005", event_type=EventType.TOOL_CALL, tool_name="eda_analysis"),
    ]
