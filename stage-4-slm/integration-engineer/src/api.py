"""
Production FastAPI Clinical Decision Support Service.
Provides offline, local clinical SLM endpoints:
- GET  /health
- GET  /model-info
- POST /summarize
- GET  /metrics
- GET  /audit/{inference_id}
- Static UI mounting for offline frontend.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import load_config, AppConfig
from model_manager import ModelManager
from safety_gateway import SafetyGateway
from audit_logger import AuditLogger
from health import HealthChecker
from inference_service import InferenceService

# Initialize configurations and components
config = load_config()
model_manager = ModelManager(
    model_path=config.model.path,
    context_length=config.model.context_length,
    threads=config.model.threads
)
safety_gateway = SafetyGateway(
    strict_mode=config.safety.strict_mode,
    confidence_threshold=config.safety.confidence_threshold
)
audit_logger = AuditLogger(log_path=config.logging.audit_log_path)
health_checker = HealthChecker(model_manager, safety_gateway, config)
inference_service = InferenceService(config, model_manager, safety_gateway, audit_logger)

# Metrics tracking
request_counter = {
    "total_requests": 0,
    "pass_count": 0,
    "review_count": 0,
    "total_latency_ms": 0.0
}

# FastAPI App
app = FastAPI(
    title="Clinical SLM Decision Support Service",
    description="Offline Local Clinical NLP Summarization and Decision-Support Assistance",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Static Frontend Directory
FRONTEND_DIR = SRC_DIR.parent / "frontend"


# --- Request & Response Models ---

class SummarizeRequest(BaseModel):
    clinical_note: str = Field(..., description="Unstructured clinical progress note or oncology summary.")
    reference_entities: Optional[Dict[str, List[str]]] = Field(None, description="Optional verified NER entities from Stage 3.")


class SummarizeResponse(BaseModel):
    inference_id: str
    timestamp: str
    risk: str
    key_finding: str
    action: str
    confidence: float
    safety_status: str
    review_required: bool
    model_version: str
    latency_ms: float
    spoken_summary: str
    provenance: Dict[str, Any]
    failure_reason: Optional[str] = None
    failed_checks: Optional[List[str]] = None
    review_status: Optional[str] = None


# --- Endpoints ---

@app.get("/health")
def get_health():
    """Returns deep health diagnostics and offline operation status."""
    return health_checker.check_health()


@app.get("/model-info")
def get_model_info():
    """Returns frozen baseline model, adapter, and quantization metadata."""
    manifest_path = SRC_DIR.parent / "artifacts" / "model_manifest.json"
    if manifest_path.exists():
        import json
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "model_name": config.model.name,
        "quantization": config.model.quantization,
        "runtime": "llama.cpp",
        "calibrated_threshold": config.safety.confidence_threshold,
        "offline_mode": config.service.offline_mode
    }


@app.post("/summarize", response_model=SummarizeResponse)
def post_summarize(payload: SummarizeRequest):
    """
    Ingests a clinical note, performs local inference, enforces the 6-stage safety firewall,
    and returns a structured decision-support briefing with voice-ready summary.
    """
    note = payload.clinical_note
    if not note or not note.strip():
        raise HTTPException(status_code=400, detail="Clinical note cannot be empty.")

    t0 = time.time()
    result = inference_service.process_note(note, reference_entities=payload.reference_entities)
    elapsed = (time.time() - t0) * 1000.0

    # Update service metrics
    request_counter["total_requests"] += 1
    if result["safety_status"] == "PASS":
        request_counter["pass_count"] += 1
    else:
        request_counter["review_count"] += 1
    request_counter["total_latency_ms"] += elapsed

    return result


@app.get("/metrics")
def get_metrics():
    """Returns cumulative operational throughput and safety statistics."""
    total = max(1, request_counter["total_requests"])
    return {
        "total_inferences": request_counter["total_requests"],
        "safety_firewall_pass_rate": round(request_counter["pass_count"] / total, 4),
        "human_review_routing_rate": round(request_counter["review_count"] / total, 4),
        "mean_latency_ms": round(request_counter["total_latency_ms"] / total, 2),
        "offline_mode": config.service.offline_mode,
        "calibrated_threshold": config.safety.confidence_threshold
    }


@app.get("/audit/{inference_id}")
def get_audit(inference_id: str):
    """Retrieves an immutable audit log record for a specified inference transaction."""
    record = audit_logger.get_record(inference_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Audit record not found for {inference_id}")
    return record


# Mount Frontend if present
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_index():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "Frontend not found."}


if __name__ == "__main__":
    import uvicorn
    print(f"Starting Clinical SLM Service on http://{config.service.host}:{config.service.port} (Offline Mode: {config.service.offline_mode})")
    uvicorn.run(app, host=config.service.host, port=config.service.port)
