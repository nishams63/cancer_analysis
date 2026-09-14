"""Tests handling failures in workflow generation, agent execution, and tools."""
from orchestration.pipeline import AADAIntegrationPipeline
from orchestration.lifecycle import RunLifecycleState
from unittest.mock import MagicMock


def test_workflow_generation_failure():
    mock_planner = MagicMock()
    mock_planner.plan.side_effect = ValueError("Decomposition cycle detected")

    pipeline = AADAIntegrationPipeline(planner=mock_planner)
    session = pipeline.run_pipeline(goal="Invalid circular goal", custom_run_id="RUN-FAIL-001")

    assert session.lifecycle_status == RunLifecycleState.FAILED
    assert session.error is not None
    assert session.error["code"] == "WORKFLOW_GENERATION_FAILED"


def test_agent_execution_failure():
    mock_executor = MagicMock()
    mock_executor.run.side_effect = RuntimeError("Tool execution crash")

    pipeline = AADAIntegrationPipeline(executor=mock_executor)
    session = pipeline.run_pipeline(goal="Analyze why revenue decreased", custom_run_id="RUN-FAIL-002")

    assert session.lifecycle_status == RunLifecycleState.FAILED
    assert session.error is not None
    assert session.error["code"] == "AGENT_EXECUTION_ERROR"
