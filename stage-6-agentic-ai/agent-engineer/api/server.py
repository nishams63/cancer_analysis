"""FastAPI REST API for AADA Agent Engineer execution and observability."""
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project paths are in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
STAGE6_DIR = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(STAGE6_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE6_DIR))

from schemas.result import AgentResult
from schemas.agent import RunStatus
from agent.executor import AgentExecutor
from trace.repository import TraceRepository
from workflow_engineer.workflows.registry import WorkflowRegistry

app = FastAPI(
    title="AADA Agent Engineer Service",
    version="1.0.0",
    description="Execution, ReAct orchestration, and observability REST API for Autonomous AI Data Analyst.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = AgentExecutor()
registry = WorkflowRegistry()
trace_repo = TraceRepository()


class RunRequest(BaseModel):
    workflow_id: Optional[str] = Field(None, description="Target workflow ID")
    intent: Optional[str] = Field(None, description="Analytical intent keyword")
    input_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Execution parameters")


class OverrideRequest(BaseModel):
    decision: str = Field("approve", description="Human decision: approve, reject, terminate")
    reason: str = Field("Analyst approval", description="Auditable rationale")


@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "AADA Agent Engineer",
        "registered_tools_count": len(executor.registry.list_tools()),
    }


@app.post("/agent/run", response_model=AgentResult)
def run_workflow(req: RunRequest) -> AgentResult:
    wf = None
    if req.workflow_id:
        wf = registry.get_template(req.workflow_id)
    if not wf and req.intent:
        wf = registry.get_template_for_intent(req.intent)
    if not wf and req.workflow_id:
        wf = registry.get_template_for_intent(req.workflow_id)

    if not wf:
        raise HTTPException(status_code=404, detail=f"Workflow template not found for input: {req}")

    try:
        result = executor.run(wf, input_data=req.input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")


@app.post("/agent/resume/{run_id}", response_model=AgentResult)
def resume_workflow(run_id: str, req: OverrideRequest) -> AgentResult:
    run_data = trace_repo.get_run(run_id)
    if not run_data:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    wf = registry.get_template(run_data["workflow_id"])
    if not wf:
        raise HTTPException(status_code=404, detail=f"Workflow '{run_data['workflow_id']}' not found.")

    try:
        res = executor.resume_after_approval(run_id=run_id, workflow=wf, decision=req.decision, reason=req.reason)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resume error: {str(e)}")


@app.post("/agent/runs/{run_id}/override")
def record_override(run_id: str, req: OverrideRequest) -> Dict[str, str]:
    run_data = trace_repo.get_run(run_id)
    if not run_data:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    now_iso = run_data.get("completed_at") or "2026-01-01T00:00:00Z"
    trace_repo.record_human_override(
        run_id=run_id,
        task_id=run_data.get("state", {}).get("current_task_id", "WORKFLOW"),
        decision=req.decision,
        reason=req.reason,
        timestamp=now_iso,
    )
    return {"status": "success", "message": f"Human override recorded for run {run_id}."}


@app.get("/agent/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    run_data = trace_repo.get_run(run_id)
    if not run_data:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return run_data


@app.get("/agent/runs/{run_id}/trace")
def get_trace(run_id: str) -> List[Dict[str, Any]]:
    events = trace_repo.get_trace(run_id)
    return [e.model_dump() for e in events]


@app.get("/agent/runs/{run_id}/status")
def get_status(run_id: str) -> Dict[str, Any]:
    run_data = trace_repo.get_run(run_id)
    if not run_data:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return {
        "run_id": run_id,
        "workflow_id": run_data["workflow_id"],
        "status": run_data["status"],
        "metrics": run_data["metrics"],
    }
