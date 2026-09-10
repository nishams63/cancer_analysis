"""
Production FastAPI Application for Stage 4 SLM Clinical Decision Support Service.
Provides low-latency REST endpoints for oncology risk, key findings, and action generation.
"""

import time
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    HealthResponse,
    ClinicalNoteRequest,
    RiskRequest,
    RiskResponse,
    ActionRequest,
    ActionResponse,
    DecisionSupportResponse,
    BatchDecisionSupportRequest,
    BatchDecisionSupportResponse
)
from service import ClinicalDecisionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("stage4.integration.app")

app = FastAPI(
    title="Personalized Precision Medicine — Stage 4 SLM Decision Support API",
    description="Production REST API serving fine-tuned clinical Small Language Models (SLM) for oncology triage and treatment optimization.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service
service = ClinicalDecisionService()


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def get_health():
    """Returns operational status, active model version, compute hardware, and uptime."""
    return service.get_health()


@app.post("/v1/generate/risk", response_model=RiskResponse, tags=["Inference"])
def generate_risk(req: RiskRequest):
    """Generates risk assessment and hazard tier from clinical notes."""
    try:
        return service.process_risk(patient_id=req.patient_id, clinical_note=req.clinical_note)
    except Exception as e:
        logger.error(f"Risk generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/generate/action", response_model=ActionResponse, tags=["Inference"])
def generate_action(req: ActionRequest):
    """Generates therapeutic action and dose modification recommendations."""
    try:
        return service.process_action(patient_id=req.patient_id, clinical_note=req.clinical_note)
    except Exception as e:
        logger.error(f"Action generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/generate/decision-support", response_model=DecisionSupportResponse, tags=["Inference"])
def generate_decision_support(req: ClinicalNoteRequest):
    """
    Primary endpoint generating the complete clinical decision support triad:
    Risk, Key Finding, and Action, verified by clinical safety guardrails.
    """
    try:
        return service.process_decision_support(
            document_id=req.document_id,
            patient_id=req.patient_id,
            clinical_note=req.clinical_note
        )
    except Exception as e:
        logger.error(f"Decision support generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/batch", response_model=BatchDecisionSupportResponse, tags=["Inference"])
def generate_batch(req: BatchDecisionSupportRequest):
    """High-throughput batch inference endpoint for clinical decision support."""
    try:
        return service.process_batch(req.notes)
    except Exception as e:
        logger.error(f"Batch generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
