"""FastAPI route handlers for AADA Integration API."""
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from .schemas import (
    StartRunRequest,
    StartRunResponse,
    RunStatusResponse,
    HumanDecisionRequest,
    HumanOverrideRequest,
    UnifiedRunResponse,
)
from .errors import AADAError
from orchestration.session_manager import get_session_manager, AADARunSession
from orchestration.coordinator import get_global_coordinator
from orchestration.pipeline import AADAIntegrationPipeline
from streaming.sse import sse_event_generator
from streaming.websocket import get_websocket_manager
from presentation.result_formatter import ResultFormatter
from presentation.trace_formatter import TraceFormatter
from presentation.evaluation_formatter import EvaluationFormatter
from human_review.approval_manager import ApprovalManager
from human_review.override_manager import OverrideManager
from agent_engineer.trace.repository import TraceRepository

router = APIRouter(prefix="/api/v1/aada", tags=["AADA Integration"])

session_manager = get_session_manager()
coordinator = get_global_coordinator()
pipeline = coordinator.pipeline
approval_mgr = ApprovalManager()
override_mgr = OverrideManager()
trace_repo = TraceRepository()
ws_manager = get_websocket_manager()


def _get_existing_session(run_id: str) -> AADARunSession:
    session = session_manager.get_session(run_id)
    if not session:
        raise AADAError(
            code="RUN_NOT_FOUND",
            message=f"Run session '{run_id}' not found.",
            run_id=run_id,
            status_code=404,
        )
    return session


@router.post("/run", response_model=StartRunResponse)
def start_run(req: StartRunRequest) -> StartRunResponse:
    """Submit a high-level user goal and initiate analysis."""
    if not req.goal or len(req.goal.strip()) < 3:
        raise AADAError(code="INVALID_GOAL", message="Goal must be at least 3 characters long.", status_code=400)

    if req.sync:
        session = pipeline.run_pipeline(goal=req.goal)
        return StartRunResponse(run_id=session.run_id, status=session.lifecycle_status.value)
    else:
        run_id = coordinator.start_run_async(goal=req.goal)
        return StartRunResponse(run_id=run_id, status="RUNNING")


@router.get("/runs/{run_id}", response_model=UnifiedRunResponse)
def get_run_full(run_id: str) -> UnifiedRunResponse:
    """Return complete integrated run state across all 4 stages."""
    session = _get_existing_session(run_id)
    events = trace_repo.get_trace(run_id)

    # Format result using presentation layer
    from agent_engineer.schemas.result import AgentResult
    agent_res = None
    if session.findings or session.evidence or session.recommendations:
        agent_res = AgentResult(
            run_id=session.run_id,
            workflow_id=session.workflow_id or "WF-UNKNOWN",
            status=session.agent_status or "completed",
            objective=session.goal,
            findings=session.findings,
            evidence=session.evidence,
            recommendations=session.recommendations,
            confidence=session.confidence or 0.0,
            completed_tasks=session.tasks_completed,
            trace_id=f"TR-{session.run_id}",
        )

    formatted_analysis = ResultFormatter.format(agent_res, goal=session.goal)

    return UnifiedRunResponse(
        run_id=session.run_id,
        status=session.lifecycle_status.value,
        goal=session.goal,
        workflow={
            "workflow_id": session.workflow_id,
            "tasks_total": session.tasks_total,
            "tasks_completed": len(session.tasks_completed),
            "metadata": session.workflow_metadata,
        },
        analysis=formatted_analysis,
        confidence=session.confidence,
        evaluation={
            "status": session.evaluation_status,
            "passed": session.evaluation_passed,
            "process_score": session.process_score,
            "outcome_score": session.outcome_score,
            "overall_score": session.overall_score,
            "details": session.evaluation_details,
        },
        human_review={
            "required": session.human_review.required,
            "status": session.human_review.status,
            "reason": session.human_review.reason,
            "current_task": session.human_review.task_id,
            "options": session.human_review.options,
        },
        trace={"event_count": len(events)},
    )


@router.get("/runs/{run_id}/status", response_model=RunStatusResponse)
def get_run_status(run_id: str) -> RunStatusResponse:
    """Quick polling endpoint for run lifecycle and progress."""
    session = _get_existing_session(run_id)
    esc_status = "REQUIRED" if session.human_review.required and session.human_review.status == "PENDING" else "NONE"
    return RunStatusResponse(
        run_id=session.run_id,
        status=session.lifecycle_status.value,
        current_task=session.current_task or (session.tasks_completed[-1] if session.tasks_completed else None),
        completed_tasks=session.tasks_completed,
        failed_tasks=session.tasks_failed,
        confidence=session.confidence,
        escalation_status=esc_status,
        progress={
            "total_tasks": session.tasks_total,
            "completed": len(session.tasks_completed),
            "percent": int((len(session.tasks_completed) / session.tasks_total * 100)) if session.tasks_total > 0 else 0,
        },
    )


@router.get("/runs/{run_id}/workflow")
def get_run_workflow(run_id: str) -> Dict[str, Any]:
    session = _get_existing_session(run_id)
    return {
        "workflow_id": session.workflow_id,
        "tasks_total": session.tasks_total,
        "metadata": session.workflow_metadata,
    }


@router.get("/runs/{run_id}/trace")
def get_run_trace(run_id: str) -> List[Dict[str, Any]]:
    """Return sanitized execution trace timeline."""
    _get_existing_session(run_id)
    events = trace_repo.get_trace(run_id)
    return TraceFormatter.format_trace(events)


@router.get("/runs/{run_id}/result")
def get_run_result(run_id: str) -> Dict[str, Any]:
    session = _get_existing_session(run_id)
    from agent_engineer.schemas.result import AgentResult
    agent_res = None
    if session.findings or session.evidence or session.recommendations:
        agent_res = AgentResult(
            run_id=session.run_id,
            workflow_id=session.workflow_id or "WF-UNKNOWN",
            status=session.agent_status or "completed",
            objective=session.goal,
            findings=session.findings,
            evidence=session.evidence,
            recommendations=session.recommendations,
            confidence=session.confidence or 0.0,
            completed_tasks=session.tasks_completed,
            trace_id=f"TR-{session.run_id}",
        )
    return ResultFormatter.format(agent_res, goal=session.goal)


@router.get("/runs/{run_id}/evaluation")
def get_run_evaluation(run_id: str) -> Dict[str, Any]:
    session = _get_existing_session(run_id)
    return {
        "status": session.evaluation_status,
        "passed": session.evaluation_passed,
        "process_score": session.process_score,
        "outcome_score": session.outcome_score,
        "overall_score": session.overall_score,
        "details": session.evaluation_details,
    }


@router.post("/runs/{run_id}/approve")
def approve_run(run_id: str, req: HumanDecisionRequest) -> Dict[str, Any]:
    """Analyst approves paused task execution and resumes pipeline."""
    session = _get_existing_session(run_id)
    if not session.human_review.required or session.human_review.status != "PENDING":
        raise AADAError(code="NO_PENDING_REVIEW", message="Run is not waiting for human approval.", run_id=run_id, status_code=400)

    # Re-plan/load workflow
    workflow = pipeline.planner.plan(session.goal)
    res = approval_mgr.approve(run_id=run_id, workflow=workflow, reason=req.reason)
    return {"status": "resumed", "run_id": run_id, "agent_status": res.status}


@router.post("/runs/{run_id}/reject")
def reject_run(run_id: str, req: HumanDecisionRequest) -> Dict[str, Any]:
    """Analyst rejects paused task execution and terminates run."""
    session = _get_existing_session(run_id)
    if not session.human_review.required or session.human_review.status != "PENDING":
        raise AADAError(code="NO_PENDING_REVIEW", message="Run is not waiting for human review.", run_id=run_id, status_code=400)

    workflow = pipeline.planner.plan(session.goal)
    res = approval_mgr.reject(run_id=run_id, workflow=workflow, reason=req.reason)
    return {"status": "rejected", "run_id": run_id, "agent_status": res.status}


@router.post("/runs/{run_id}/override")
def override_run(run_id: str, req: HumanOverrideRequest) -> Dict[str, Any]:
    """Analyst provides explicit override decision and resumes execution."""
    session = _get_existing_session(run_id)
    if not session.human_review.required or session.human_review.status != "PENDING":
        raise AADAError(code="NO_PENDING_REVIEW", message="Run is not waiting for human review.", run_id=run_id, status_code=400)

    workflow = pipeline.planner.plan(session.goal)
    res = override_mgr.submit_override(
        run_id=run_id,
        workflow=workflow,
        decision=req.decision,
        reason=req.reason,
        override_parameters=req.override_parameters,
    )
    return {"status": "overridden", "run_id": run_id, "agent_status": res.status}


@router.get("/runs/{run_id}/stream")
async def stream_run_events(run_id: str):
    """Server-Sent Events stream for real-time live execution updates."""
    _get_existing_session(run_id)
    return StreamingResponse(
        sse_event_generator(run_id=run_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.websocket("/runs/{run_id}/ws")
async def websocket_run_trace(websocket: WebSocket, run_id: str):
    """WebSocket connection for real-time trace events."""
    await ws_manager.connect(run_id, websocket)
    try:
        while True:
            # Wait for any incoming client messages (e.g., ping)
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(run_id, websocket)
