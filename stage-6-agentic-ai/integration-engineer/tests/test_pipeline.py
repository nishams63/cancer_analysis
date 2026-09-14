"""Tests for AADAIntegrationPipeline executing through all 4 stages."""
from orchestration.pipeline import AADAIntegrationPipeline
from orchestration.lifecycle import RunLifecycleState


def test_end_to_end_pipeline(pipeline, session_manager):
    goal = "Analyze why revenue decreased last quarter"
    session = pipeline.run_pipeline(goal=goal, custom_run_id="RUN-TEST-001")

    assert session.run_id == "RUN-TEST-001"
    assert session.lifecycle_status in (RunLifecycleState.COMPLETED, RunLifecycleState.WAITING_FOR_HUMAN)
    assert session.workflow_id is not None
    assert session.tasks_total > 0
    assert len(session.tasks_completed) > 0
    assert session.confidence is not None


def test_goal_submission_and_workflow_integration(pipeline):
    goal = "Investigate anomalous transaction volume spikes"
    session = pipeline.run_pipeline(goal=goal, auto_evaluate=False)

    assert session.workflow_id is not None
    assert "anomaly" in session.workflow_id.lower() or "anom" in session.workflow_id.lower() or session.tasks_total > 0
    assert len(session.tasks_completed) > 0


def test_evaluation_integration(pipeline):
    goal = "Analyze customer churn risk factors"
    session = pipeline.run_pipeline(goal=goal, auto_evaluate=True)

    assert session.evaluation_status == "COMPLETED"
    assert session.overall_score is not None
    assert session.process_score is not None
    assert session.outcome_score is not None
    assert session.evaluation_passed is not None
