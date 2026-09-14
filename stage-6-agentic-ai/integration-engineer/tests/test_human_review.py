"""Tests for Human-in-the-loop pause, approval, rejection, and override."""
from orchestration.session_manager import AADARunSession, get_session_manager
from orchestration.lifecycle import RunLifecycleState
from human_review.approval_manager import ApprovalManager
from human_review.override_manager import OverrideManager
from workflow_engineer.planner.workflow_planner import WorkflowPlanner


def test_human_pause_and_approval():
    session_mgr = get_session_manager()
    session = session_mgr.create_session(run_id="RUN-PAUSE-001", goal="Analyze why revenue declined")
    session.update_status(RunLifecycleState.WAITING_FOR_HUMAN)
    session.human_review.required = True
    session.human_review.status = "PENDING"
    session.human_review.reason = "Ambiguous root cause candidates"
    session.current_task = "T005"
    session_mgr.save_session(session)

    planner = WorkflowPlanner()
    wf = planner.plan("Analyze why revenue declined")

    approval_mgr = ApprovalManager()
    res = approval_mgr.approve(run_id="RUN-PAUSE-001", workflow=wf, reason="Approved by senior analyst")

    updated_session = session_mgr.get_session("RUN-PAUSE-001")
    assert updated_session.human_review.status == "APPROVED"
    assert res is not None


def test_human_rejection():
    session_mgr = get_session_manager()
    session = session_mgr.create_session(run_id="RUN-REJECT-001", goal="Analyze why revenue declined")
    session.update_status(RunLifecycleState.WAITING_FOR_HUMAN)
    session.human_review.required = True
    session.human_review.status = "PENDING"
    session_mgr.save_session(session)

    planner = WorkflowPlanner()
    wf = planner.plan("Analyze why revenue declined")

    approval_mgr = ApprovalManager()
    res = approval_mgr.reject(run_id="RUN-REJECT-001", workflow=wf, reason="High risk of erroneous action")

    updated_session = session_mgr.get_session("RUN-REJECT-001")
    assert updated_session.human_review.status == "REJECTED"
    assert updated_session.lifecycle_status == RunLifecycleState.CANCELLED


def test_human_override():
    session_mgr = get_session_manager()
    session = session_mgr.create_session(run_id="RUN-OVERRIDE-001", goal="Analyze why revenue declined")
    session.update_status(RunLifecycleState.WAITING_FOR_HUMAN)
    session.human_review.required = True
    session.human_review.status = "PENDING"
    session_mgr.save_session(session)

    planner = WorkflowPlanner()
    wf = planner.plan("Analyze why revenue declined")

    override_mgr = OverrideManager()
    res = override_mgr.submit_override(
        run_id="RUN-OVERRIDE-001",
        workflow=wf,
        decision="Select Regional Hypothesis",
        reason="Analyst confirmed external marketing campaign impact",
    )

    updated_session = session_mgr.get_session("RUN-OVERRIDE-001")
    assert updated_session.human_review.status == "OVERRIDDEN"
