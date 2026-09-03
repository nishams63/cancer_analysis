"""
FastAPI Inference Service for Stage 1 ML Oncology Toxicity Risk Prediction.
Role: Integration Engineer.

Endpoints:
- GET  /health         : Health check verifying frozen V4 artifact availability
- POST /predict        : Single patient encounter toxicity prediction
- POST /predict/batch  : Batch patient encounter toxicity prediction
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from schemas import (
    PatientToxicityInput,
    ToxicityPredictionResponse,
    BatchToxicityPredictionResponse,
    HealthResponse
)
from predictor import V4InferenceEngine


# Global engine holder
engine: V4InferenceEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and verifies the frozen V4 model on service startup."""
    global engine
    try:
        engine = V4InferenceEngine()
    except Exception as exc:
        print(f"CRITICAL: Failed to load V4 model artifacts during startup: {exc}")
        engine = None
    yield
    engine = None


app = FastAPI(
    title="Oncology Toxicity Risk Prediction Service",
    description=(
        "Research decision-support API serving the frozen Stage 1 Candidate V4 "
        "machine learning model for oncology adverse event toxicity risk classification."
    ),
    version="1.0.0",
    lifespan=lifespan
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Returns clean, user-friendly JSON error messages when required fields are missing
    or invalid data types are supplied.
    """
    errors = []
    for err in exc.errors():
        field_loc = " -> ".join([str(loc) for loc in err["loc"] if loc != "body"])
        msg = err["msg"]
        errors.append(f"Field '{field_loc}': {msg}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Input validation failed",
            "details": errors
        }
    )


@app.get("/", tags=["Info"])
async def root():
    """Service metadata and documentation pointer."""
    return {
        "service": "Oncology Toxicity Risk Prediction Service",
        "stage": "Stage 1 ML",
        "model_version": "V4",
        "status": "online" if engine and engine.is_healthy() else "degraded",
        "docs_url": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """
    Health check endpoint verifying that all frozen V4 artifacts are loaded in memory.
    """
    if engine is None or not engine.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded or engine is unhealthy."
        )

    return HealthResponse(
        status="ok",
        model_version=engine.model_version,
        artifacts_loaded=True,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


@app.post(
    "/predict",
    response_model=ToxicityPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"]
)
async def predict_single_patient(patient_input: PatientToxicityInput):
    """
    Evaluates toxicity risk for a single patient encounter.
    Accepts all 30 raw predictor features and computes engineered features automatically.
    Returns: 'Low', 'Moderate', or 'High' with genuine calibrated class probabilities.
    """
    if engine is None or not engine.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not initialized. Check server logs."
        )

    try:
        result = engine.predict_single(patient_input.model_dump())
        return ToxicityPredictionResponse(**result)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}"
        )


@app.post(
    "/predict/batch",
    response_model=BatchToxicityPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"]
)
async def predict_batch_patients(patient_inputs: List[PatientToxicityInput]):
    """
    Evaluates toxicity risk for a batch of patient encounters.
    """
    if engine is None or not engine.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not initialized. Check server logs."
        )

    if not patient_inputs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch list must contain at least one patient record."
        )

    try:
        raw_dicts = [p.model_dump() for p in patient_inputs]
        batch_results = engine.predict_batch(raw_dicts)
        return BatchToxicityPredictionResponse(
            predictions=[ToxicityPredictionResponse(**r) for r in batch_results],
            total_records=len(batch_results),
            model_version=engine.model_version
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(exc)}"
        )
