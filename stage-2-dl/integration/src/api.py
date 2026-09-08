"""
Stage 2 Deep Learning - Multimodal Oncology Patient-Level Inference Service (FastAPI)

Exposes REST endpoints:
  - GET  /health                   System health, readiness, and loaded model registry status
  - GET  /info                     Model configurations, alert thresholds, disclaimers
  - POST /predict/patient          Single patient multimodal inference
  - POST /predict/batch            Batch inference with per-patient error isolation
  - POST /predict/sample/{patient_id} Demo endpoint loading sample development cohort cases

MANDATORY REGULATORY NOTICE:
Research prototype developed with synthetic data. NOT clinically validated.
Performance does not establish clinical safety or efficacy.
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

STAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STAGE_ROOT / '.runtime'))
sys.path.insert(0, str(STAGE_ROOT))

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.staticfiles import StaticFiles
import uvicorn
import pandas as pd

try:
    from . import config, validation, schemas
    from .model_registry import ModelRegistry, MissingCheckpointError
    from .inference_service import PatientInferenceService
    from .integration_pipeline import _load_sample_patient_data
except (ImportError, ValueError):
    import config, validation, schemas
    from model_registry import ModelRegistry, MissingCheckpointError
    from inference_service import PatientInferenceService
    from integration_pipeline import _load_sample_patient_data

from dl.research_inference import ResearchPredictor


app = FastAPI(
    title="Multimodal Oncology Patient Inference Service",
    description=(
        "Patient-level multimodal inference uniting ResNet-50 Attention-MIL Pathology "
        "and Continuous Temporal Transformer with Epistemic Uncertainty and OOD Detection. "
        "MANDATORY: Research prototype developed with synthetic data. NOT clinically validated."
    ),
    version="2.1.0-upgraded"
)

# Mount pathology tiles static directory for rich browser visualization
tiles_dir = config.STAGE_2_DIR / "data-engineering" / "data" / "v2" / "processed" / "pathology_tiles"
if tiles_dir.exists():
    app.mount("/tiles", StaticFiles(directory=str(tiles_dir)), name="tiles")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
def visual_dashboard():
    """Serves the interactive visual multimodal oncology dashboard."""
    dash_path = Path(__file__).resolve().parent.parent / "dashboard" / "dashboard.html"
    if dash_path.exists():
        with open(dash_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return RedirectResponse(url="/docs")


@app.get("/viewer/{patient_id}", response_class=HTMLResponse, tags=["Viewer"])
def view_patient(patient_id: str):
    """Renders self-contained interactive HTML dashboard report for a patient."""
    try:
        from integration.dashboard.standalone_viewer import generate_patient_html_report
        path = generate_patient_html_report(patient_id)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate viewer: {str(e)}")


@app.get("/health", tags=["System"])
def health_check():
    """
    Returns system readiness, loaded model status, and mandatory regulatory disclaimers.
    Reports 'degraded' if critical models are unavailable.
    """
    registry_status = ModelRegistry.status_summary()
    models_dict = {k: "available" if v["available"] else "missing" for k, v in registry_status.items()}

    if ResearchPredictor.ready():
        return {
            'status': 'healthy',
            'service': 'stage-2-multimodal-integration',
            'models': models_dict,
            'validated_artifacts_ready': True,
            'selected_configuration': ResearchPredictor.shared().selected,
            'mandatory_disclaimer': config.MANDATORY_DISCLAIMER,
            'equivalence_disclaimer': config.EQUIVALENCE_DISCLAIMER
        }
    has_pathology = registry_status.get("pathology_resnet50", {}).get("available", False) or \
                    registry_status.get("baseline_pathology", {}).get("available", False)
    has_temporal = registry_status.get("temporal_transformer", {}).get("available", False) or \
                   registry_status.get("baseline_temporal", {}).get("available", False)

    system_status = "healthy" if (has_pathology and has_temporal) else "degraded"

    return {
        "status": system_status,
        "service": "stage-2-multimodal-integration",
        "version": "stage2-upgraded-v1",
        "models": {k: "available" if v["available"] else "missing" for k, v in registry_status.items()},
        "mandatory_disclaimer": config.MANDATORY_DISCLAIMER,
        "equivalence_disclaimer": config.EQUIVALENCE_DISCLAIMER
    }


@app.get("/info", tags=["System"])
def service_info():
    """Returns configuration parameters, alert thresholds, and disclaimers."""
    return {
        "service_version": "stage2-upgraded-v1",
        "data_source": "synthetic_only",
        "clinical_validation": "NONE",
        "supported_fusion_modes": ["gated", "concat_mlp", "cross_attention", "fixed_weighted"],
        "supported_aggregations": ["attention", "mean", "median", "max"],
        "historical_cutoff_day": config.FORECAST_SPLIT_DAY,
        "prototype_alert_thresholds": config.PROTOTYPE_ALERT_THRESHOLDS,
        "prototype_alert_labels": config.PROTOTYPE_ALERT_LABELS,
        "mandatory_disclaimer": config.MANDATORY_DISCLAIMER,
        "equivalence_disclaimer": config.EQUIVALENCE_DISCLAIMER
    }


@app.post("/predict/patient", tags=["Inference"])
def predict_patient(request: schemas.PatientInferenceRequest):
    """
    Executes unified patient-level inference combining pathology tiles and longitudinal biomarkers.
    Enforces historical cutoff Day <= 90 and returns calibrated risk with epistemic uncertainty.
    """
    try:
        use_validated = (request.model_version == 'validated') or (request.model_configuration == 'validated')
        if use_validated:
            if not ResearchPredictor.ready():
                raise HTTPException(status_code=503, detail='Validated experiments have not completed')
            res = ResearchPredictor.shared().predict(
                request.patient_id,
                request.tile_paths,
                [obs.model_dump() for obs in request.temporal_observations] if request.temporal_observations else None
            )
            if 'inference_mode' not in res:
                res['inference_mode'] = res.get('modality_status')
            if 'risk' not in res:
                lvl = res.get('risk_level') or 'LOW'
                res['risk'] = {'level': lvl, 'joint_confidence_status': 'INTERMEDIATE', 'description': f'Model risk: {lvl}'}
            return res

        if (request.model_version == 'auto' or request.model_configuration == 'auto') and ResearchPredictor.ready():
            try:
                res = ResearchPredictor.shared().predict(
                    request.patient_id,
                    request.tile_paths,
                    [obs.model_dump() for obs in request.temporal_observations] if request.temporal_observations else None
                )
                if 'inference_mode' not in res:
                    res['inference_mode'] = res.get('modality_status')
                if 'risk' not in res:
                    lvl = res.get('risk_level') or 'LOW'
                    res['risk'] = {'level': lvl, 'joint_confidence_status': 'INTERMEDIATE', 'description': f'Model risk: {lvl}'}
                return res
            except Exception:
                pass

        service = PatientInferenceService.get_shared()
        res = service.predict(request)
        res_dict = res.model_dump()
        res_dict['modality_status'] = res.inference_mode
        return res_dict

    except HTTPException:
        raise
    except validation.ValidationError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except MissingCheckpointError as mce:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(mce))
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failure: {str(e)}"
        )


@app.post("/predict/batch", tags=["Inference"])
def predict_batch(request: schemas.BatchInferenceRequest):
    """
    Processes a batch of patients with isolated per-patient error handling.
    Malformed patient records do not prevent valid patients from completing.
    """
    results: List[Dict[str, Any]] = []
    successful = 0
    failed = 0

    for pt_request in request.patients:
        try:
            prediction = predict_patient(pt_request)
            results.append({
                "patient_id": pt_request.patient_id,
                "status": "success",
                "status_code": 200,
                "detail": None,
                "prediction": prediction
            })
            successful += 1
        except HTTPException as exc:
            results.append({
                "patient_id": pt_request.patient_id,
                "status": "error",
                "status_code": exc.status_code,
                "detail": exc.detail,
                "prediction": None
            })
            failed += 1
        except validation.ValidationError as ve:
            results.append({
                "patient_id": pt_request.patient_id,
                "status": "error",
                "status_code": 400,
                "detail": f"Validation error: {str(ve)}",
                "prediction": None
            })
            failed += 1
        except MissingCheckpointError as mce:
            results.append({
                "patient_id": pt_request.patient_id,
                "status": "error",
                "status_code": 503,
                "detail": f"Checkpoint error: {str(mce)}",
                "prediction": None
            })
            failed += 1
        except Exception as e:
            results.append({
                "patient_id": pt_request.patient_id,
                "status": "error",
                "status_code": 500,
                "detail": f"Processing error: {str(e)}",
                "prediction": None
            })
            failed += 1

    return {
        "count": len(results),
        "total": len(results),
        "successful": successful,
        "failed": failed,
        "results": results
    }


@app.get("/predict/sample/{patient_id}", tags=["Demo"])
@app.post("/predict/sample/{patient_id}", tags=["Demo"])
def predict_sample_patient(
    patient_id: str,
    aggregation: str = "attention",
    fusion: str = "gated",
    model_config: str = "auto"
):
    """
    Demo endpoint: Loads sample data for a patient from development data and runs inference.
    """
    tiles, bio_df = _load_sample_patient_data(patient_id)
    if not tiles and (bio_df is None or len(bio_df) == 0):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient '{patient_id}' not found in sample dataset."
        )

    obs_list = []
    if bio_df is not None and len(bio_df) > 0:
        for _, row in bio_df.iterrows():
            obs_list.append(schemas.BiomarkerObservation(
                days_from_baseline=int(row['days_from_baseline']),
                ctDNA_vaf_percent=float(row['ctDNA_vaf_percent']) if pd.notna(row.get('ctDNA_vaf_percent')) else None,
                cea_ng_ml=float(row['cea_ng_ml']) if pd.notna(row.get('cea_ng_ml')) else None,
                ca125_u_ml=float(row['ca125_u_ml']) if pd.notna(row.get('ca125_u_ml')) else None,
                ldh_u_l=float(row['ldh_u_l']) if pd.notna(row.get('ldh_u_l')) else None,
                crp_mg_l=float(row['crp_mg_l']) if pd.notna(row.get('crp_mg_l')) else None,
                delta_days=float(row['delta_days']) if pd.notna(row.get('delta_days')) else None,
            ))

    req = schemas.PatientInferenceRequest(
        patient_id=patient_id,
        tile_paths=tiles if tiles else None,
        temporal_observations=obs_list if obs_list else None,
        model_configuration=model_config,
        fusion_mode=fusion,
        aggregation_method=aggregation
    )

    res = predict_patient(req)
    if hasattr(res, "model_dump"):
        res_dict = res.model_dump()
    elif isinstance(res, dict):
        res_dict = dict(res)
    else:
        res_dict = {"prediction": res}

    # Format direct static URLs and classes for all biopsy tiles
    tile_urls = []
    tile_classes = []
    for tp in (tiles or []):
        p = Path(tp)
        cls_name = p.parent.name
        tile_urls.append(f"/tiles/{cls_name}/{p.name}")
        tile_classes.append(cls_name)

    res_dict["tile_urls"] = tile_urls
    res_dict["tile_classes"] = tile_classes

    # Format sorted historical biomarker visits for interactive charting
    biomarkers = []
    if bio_df is not None and len(bio_df) > 0:
        for _, row in bio_df.sort_values("days_from_baseline").iterrows():
            biomarkers.append({
                "days_from_baseline": int(row["days_from_baseline"]),
                "ctDNA_vaf_percent": float(row["ctDNA_vaf_percent"]) if pd.notna(row.get("ctDNA_vaf_percent")) else None,
                "cea_ng_ml": float(row["cea_ng_ml"]) if pd.notna(row.get("cea_ng_ml")) else None,
                "ca125_u_ml": float(row["ca125_u_ml"]) if pd.notna(row.get("ca125_u_ml")) else None,
                "ldh_u_l": float(row["ldh_u_l"]) if pd.notna(row.get("ldh_u_l")) else None,
                "crp_mg_l": float(row["crp_mg_l"]) if pd.notna(row.get("crp_mg_l")) else None,
            })
    res_dict["historical_biomarkers"] = biomarkers

    # Elevate attention weights if inside pathology summary
    if "attention_weights" not in res_dict or not res_dict["attention_weights"]:
        p_info = res_dict.get("pathology")
        if isinstance(p_info, dict) and p_info.get("attention_weights"):
            res_dict["attention_weights"] = p_info["attention_weights"]

    return res_dict


def start_server(host: str = config.API_HOST, port: int = config.API_PORT):
    """Launches the Uvicorn server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
