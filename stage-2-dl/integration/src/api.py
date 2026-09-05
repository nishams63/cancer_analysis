"""
Stage 2 Deep Learning - Multimodal Inference REST API Service

FastAPI service exposing:
  - GET /health
  - GET /info
  - POST /predict/patient
  - POST /predict/sample/{patient_id}

MANDATORY NOTICE:
This API is a research prototype developed with synthetic data.
NOT clinically validated. Never use for clinical decision-making.
"""
import os
import sys
from typing import List, Dict, Any, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import uvicorn

try:
    from . import config, validation, integration_pipeline
except (ImportError, ValueError):
    import config, validation, integration_pipeline


# Initialize FastAPI App
app = FastAPI(
    title="Multimodal Oncology Inference Service (Prototype)",
    description=(
        "Research prototype inference service uniting frozen histopathology CNN and temporal BiLSTM. "
        "IMPORTANT: This service was developed using synthetic data and has not been clinically validated. "
        "Performance does not establish clinical safety or efficacy."
    ),
    version="2.0.0-prototype"
)


# Pydantic Schemas
class BiomarkerObservation(BaseModel):
    days_from_baseline: int = Field(..., ge=0, le=90, description="Elapsed days since baseline (must be <= 90)")
    ctDNA_vaf_percent: Optional[float] = Field(None, description="ctDNA variant allele frequency (%)")
    cea_ng_ml: Optional[float] = Field(None, description="Carcinoembryonic antigen (ng/mL)")
    ca125_u_ml: Optional[float] = Field(None, description="Cancer antigen 125 (U/mL)")
    ldh_u_l: Optional[float] = Field(None, description="Lactate dehydrogenase (U/L)")
    crp_mg_l: Optional[float] = Field(None, description="C-reactive protein (mg/L)")
    delta_days: Optional[int] = Field(None, description="Days since previous visit")
    ctDNA_velocity_30d: Optional[float] = Field(None, description="Backward-looking velocity")
    ctDNA_missing: Optional[int] = Field(0, description="Missingness mask")
    cea_missing: Optional[int] = Field(0, description="Missingness mask")
    ca125_missing: Optional[int] = Field(0, description="Missingness mask")
    ldh_missing: Optional[int] = Field(0, description="Missingness mask")
    crp_missing: Optional[int] = Field(0, description="Missingness mask")


class PatientInferenceRequest(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier")
    tile_paths: Optional[List[str]] = Field(None, description="List of absolute paths to biopsy tiles")
    temporal_observations: Optional[List[BiomarkerObservation]] = Field(
        None, description="Longitudinal history visits strictly within days <= 90"
    )
    aggregation_method: Optional[str] = Field("mean", description="Tile aggregation method: mean, median, max")
    fusion_method: Optional[str] = Field("weighted_linear", description="Fusion method: weighted_linear or rule_based")
    custom_weights: Optional[Dict[str, float]] = Field(None, description="Optional custom weights summing to 1.0")


@app.get("/health", tags=["System"])
def health_check():
    """Returns service health status and loaded model checkpoint information."""
    pipeline = integration_pipeline.MultimodalPatientPipeline.get_shared_pipeline()
    return {
        "status": "healthy",
        "service": "stage-2-multimodal-integration",
        "models_loaded": {
            "pathology_cnn": pipeline.pathology_checkpoint_path,
            "temporal_bilstm": pipeline.temporal_checkpoint_path
        },
        "clinical_validation_status": config.CLINICAL_VALIDATION_STATUS,
        "data_source": config.DATA_SOURCE,
        "mandatory_disclaimer": config.MANDATORY_DISCLAIMER,
        "equivalence_disclaimer": config.EQUIVALENCE_DISCLAIMER
    }


@app.get("/info", tags=["System"])
def service_info():
    """Returns default fusion configuration, alert thresholds, and disclaimers."""
    return {
        "notice": "Prototype parameters only. NOT clinically calibrated.",
        "default_fusion_weights": config.DEFAULT_FUSION_WEIGHTS,
        "temporal_only_weights": config.TEMPORAL_ONLY_WEIGHTS,
        "prototype_alert_thresholds": config.PROTOTYPE_ALERT_THRESHOLDS,
        "prototype_alert_labels": config.PROTOTYPE_ALERT_LABELS,
        "forecast_boundary_day": config.FORECAST_SPLIT_DAY,
        "disclaimer": config.MANDATORY_DISCLAIMER
    }


@app.post("/predict/patient", tags=["Inference"])
def predict_patient(request: PatientInferenceRequest):
    """
    Runs multimodal patient-level inference combining pathology tiles and longitudinal biomarkers.
    """
    try:
        # Convert observation list to DataFrame if provided
        history_df = None
        if request.temporal_observations:
            obs_dicts = [obs.dict(exclude_none=True) for obs in request.temporal_observations]
            history_df = pd.DataFrame(obs_dicts)

        pipeline = integration_pipeline.MultimodalPatientPipeline.get_shared_pipeline()
        result = pipeline.run_patient_inference(
            patient_id=request.patient_id,
            pathology_tiles=request.tile_paths,
            temporal_history=history_df,
            aggregation_method=request.aggregation_method or config.DEFAULT_TILE_AGGREGATION,
            fusion_method=request.fusion_method or 'weighted_linear',
            custom_weights=request.custom_weights
        )
        return result

    except validation.ValidationError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {str(e)}")


@app.post("/predict/sample/{patient_id}", tags=["Demo / Testing"])
def predict_sample_patient(patient_id: str, aggregation: str = "mean", fusion: str = "weighted_linear"):
    """
    Demo endpoint: loads sample data for patient_id from the development cohort and executes inference.
    """
    try:
        tiles, bio_df = integration_pipeline._load_sample_patient_data(patient_id)
        if not tiles and (bio_df is None or len(bio_df) == 0):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Patient {patient_id} not found in sample cohort.")

        pipeline = integration_pipeline.MultimodalPatientPipeline.get_shared_pipeline()
        result = pipeline.run_patient_inference(
            patient_id=patient_id,
            pathology_tiles=tiles,
            temporal_history=bio_df,
            aggregation_method=aggregation,
            fusion_method=fusion
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


def start_server(host: str = config.API_HOST, port: int = config.API_PORT):
    """Launches the Uvicorn server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
