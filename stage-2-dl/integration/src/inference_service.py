"""
Stage 2 Deep Learning - Patient Inference Orchestration Service

Central single-point-of-truth service coordinating:
  1. Input validation & anti-leakage checking
  2. Modality state resolution (FULL, PATHOLOGY_ONLY, TEMPORAL_ONLY, INSUFFICIENT_DATA)
  3. Preprocessing & feature extraction
  4. Pathology inference (ResNet-50 + Attention-MIL / ResNet-18 baseline)
  5. Temporal forecasting (Continuous Temporal Transformer / BiLSTM baseline)
  6. Multimodal fusion (Gated, Concat-MLP, Cross-Attention, Fixed)
  7. Epistemic uncertainty estimation & probability calibration
  8. Out-of-Distribution (OOD) detection
  9. Research risk stratification
  10. Latency tracking and structured audit logging
"""
import time
from typing import Dict, Any, Optional, List
import pandas as pd
import torch

try:
    from . import config, model_registry, schemas, validation, logging_utils
    from .model_registry import ModelRegistry, MissingCheckpointError
    from .pathology_service import PathologyService
    from .temporal_service import TemporalService
    from .fusion_service import FusionService
    from .uncertainty_service import UncertaintyService
    from .ood_service import OODService
    from .risk_engine import RiskEngine
except (ImportError, ValueError):
    import config, model_registry, schemas, validation, logging_utils
    from model_registry import ModelRegistry, MissingCheckpointError
    from pathology_service import PathologyService
    from temporal_service import TemporalService
    from fusion_service import FusionService
    from uncertainty_service import UncertaintyService
    from ood_service import OODService
    from risk_engine import RiskEngine


class PatientInferenceService:
    """Singleton service orchestrating patient-level multimodal inference."""
    _instance = None

    def __init__(self):
        self.pathology_service = PathologyService.get_shared()
        self.temporal_service = TemporalService.get_shared()
        self.fusion_service = FusionService.get_shared()
        self.uncertainty_service = UncertaintyService.get_shared()
        self.ood_service = OODService.get_shared()

    @classmethod
    def get_shared(cls) -> 'PatientInferenceService':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def predict(self, request: schemas.PatientInferenceRequest) -> schemas.PatientInferenceResponse:
        """
        Executes end-to-end patient inference.

        Args:
            request: Validated PatientInferenceRequest schema.

        Returns:
            Standardized PatientInferenceResponse schema.
        """
        tracker = logging_utils.LatencyTracker()
        total_start = time.perf_counter()

        # 1. Validation & Modality State
        with tracker.time_block("validation"):
            patient_id = validation.validate_patient_id(request.patient_id)
            v_tiles = validation.validate_pathology_tiles(request.tile_paths)
            
            # Convert observations to DataFrame if provided
            if request.temporal_observations:
                obs_dicts = [obs.model_dump() for obs in request.temporal_observations]
                history_df = pd.DataFrame(obs_dicts)
                v_history = validation.validate_temporal_history(history_df)
            else:
                v_history = None

            has_p = bool(v_tiles)
            has_t = bool(v_history is not None and len(v_history) > 0)
            mode = validation.determine_modality_status(has_p, has_t)

        # 2. Select Architecture Configuration
        cfg_choice = request.model_configuration
        if cfg_choice == 'baseline':
            backbone = 'resnet18'
            agg_method = request.aggregation_method if request.aggregation_method in ['mean', 'median', 'max'] else 'mean'
            temporal_arch = 'bilstm'
            fusion_mode = 'fixed_weighted'
            active_version = "stage2-baseline-r18-bilstm-v1"
        else:
            backbone = 'resnet50' if ModelRegistry.is_available('pathology_resnet50') else 'resnet18'
            agg_method = request.aggregation_method or 'attention'
            temporal_arch = 'transformer'
            fusion_mode = request.fusion_mode or 'gated'
            active_version = f"stage2-upgraded-{backbone}-{temporal_arch}-{fusion_mode}-v1"

        avail_modalities = [m for m, b in [("pathology", has_p), ("temporal", has_t)] if b]

        # 3. Handle Mode: INSUFFICIENT_DATA
        if mode == "INSUFFICIENT_DATA":
            total_ms = (time.perf_counter() - total_start) * 1000.0
            tracker.timings["total_ms"] = round(total_ms, 2)
            logging_utils.log_inference_event(
                patient_id=patient_id,
                inference_mode=mode,
                model_version=active_version,
                risk_level="INSUFFICIENT_DATA",
                ood_status="NOT_EVALUATED",
                latency_ms=total_ms
            )
            return schemas.PatientInferenceResponse(
                patient_id=patient_id,
                inference_mode=mode,
                available_modalities=[],
                model_configuration=active_version,
                progression_probability=None,
                ctdna_forecast_30d=None,
                risk=schemas.RiskOutput(
                    level="INSUFFICIENT_DATA",
                    joint_confidence_status="NOT_EVALUATED",
                    description="Insufficient data: neither biopsy tiles nor biomarker observations provided. Model abstained."
                ),
                pathology=None,
                temporal=None,
                fusion=None,
                uncertainty=schemas.UncertaintyOutput(
                    confidence=None,
                    uncertainty_score=None,
                    predictive_variance=None,
                    calibration_status="NOT_CALIBRATED"
                ),
                ood=schemas.OODOutput(status="NOT_EVALUATED"),
                latency=schemas.LatencySummary(
                    validation_ms=tracker.timings.get("validation", 0.0),
                    total_ms=round(total_ms, 2)
                ),
                system={"version": active_version, "environment": "research-prototype"},
                mandatory_disclaimer=config.MANDATORY_DISCLAIMER,
                equivalence_disclaimer=config.EQUIVALENCE_DISCLAIMER
            )

        # 4. Handle Mode: PATHOLOGY_ONLY
        if mode == "PATHOLOGY_ONLY":
            with tracker.time_block("pathology"):
                path_summary, p_repr = self.pathology_service.process_tiles(
                    v_tiles, backbone=backbone, aggregation_method=agg_method
                )

            prog_prob = path_summary.malignant_probability
            ctdna_forecast = None

            with tracker.time_block("uncertainty"):
                path_model = self.pathology_service.get_pathology_model(backbone)
                uncertainty_out = schemas.UncertaintyOutput(
                    confidence=round(max(prog_prob, 1.0 - prog_prob), 4),
                    uncertainty_score=None,
                    predictive_variance=None,
                    calibration_status="NOT_CALIBRATED"
                )

            risk_out = RiskEngine.stratify_risk(prog_prob, uncertainty=uncertainty_out.uncertainty_score)
            ood_out = schemas.OODOutput(status="NOT_EVALUATED")

            total_ms = (time.perf_counter() - total_start) * 1000.0
            tracker.timings["total_ms"] = round(total_ms, 2)

            logging_utils.log_inference_event(
                patient_id=patient_id,
                inference_mode=mode,
                model_version=active_version,
                risk_level=risk_out.level,
                ood_status=ood_out.status,
                latency_ms=total_ms
            )

            return schemas.PatientInferenceResponse(
                patient_id=patient_id,
                inference_mode=mode,
                available_modalities=avail_modalities,
                model_configuration=active_version,
                progression_probability=prog_prob,
                ctdna_forecast_30d=None,
                risk=risk_out,
                pathology=path_summary,
                temporal=None,
                fusion=None,
                uncertainty=uncertainty_out,
                ood=ood_out,
                latency=schemas.LatencySummary(
                    validation_ms=tracker.timings.get("validation", 0.0),
                    pathology_ms=tracker.timings.get("pathology", 0.0),
                    uncertainty_ms=tracker.timings.get("uncertainty", 0.0),
                    total_ms=round(total_ms, 2)
                ),
                system={"version": active_version, "environment": "research-prototype"},
                mandatory_disclaimer=config.MANDATORY_DISCLAIMER,
                equivalence_disclaimer=config.EQUIVALENCE_DISCLAIMER
            )

        # 5. Handle Mode: TEMPORAL_ONLY
        if mode == "TEMPORAL_ONLY":
            with tracker.time_block("temporal"):
                temp_summary, t_repr, t_inputs = self.temporal_service.process_observations(
                    v_history, architecture=temporal_arch
                )

            prog_prob = temp_summary.progression_probability
            ctdna_forecast = temp_summary.ctdna_30d_forecast

            with tracker.time_block("uncertainty"):
                temporal_model = self.temporal_service.get_temporal_model(temporal_arch)
                uncertainty_out = self.uncertainty_service.estimate_uncertainty(
                    temporal_model, *t_inputs, current_probability=prog_prob
                )

            risk_out = RiskEngine.stratify_risk(prog_prob, uncertainty=uncertainty_out.uncertainty_score)
            ood_out = schemas.OODOutput(status="NOT_EVALUATED")

            total_ms = (time.perf_counter() - total_start) * 1000.0
            tracker.timings["total_ms"] = round(total_ms, 2)

            logging_utils.log_inference_event(
                patient_id=patient_id,
                inference_mode=mode,
                model_version=active_version,
                risk_level=risk_out.level,
                ood_status=ood_out.status,
                latency_ms=total_ms
            )

            return schemas.PatientInferenceResponse(
                patient_id=patient_id,
                inference_mode=mode,
                available_modalities=avail_modalities,
                model_configuration=active_version,
                progression_probability=prog_prob,
                ctdna_forecast_30d=ctdna_forecast,
                risk=risk_out,
                pathology=None,
                temporal=temp_summary,
                fusion=None,
                uncertainty=uncertainty_out,
                ood=ood_out,
                latency=schemas.LatencySummary(
                    validation_ms=tracker.timings.get("validation", 0.0),
                    temporal_ms=tracker.timings.get("temporal", 0.0),
                    uncertainty_ms=tracker.timings.get("uncertainty", 0.0),
                    total_ms=round(total_ms, 2)
                ),
                system={"version": active_version, "environment": "research-prototype"},
                mandatory_disclaimer=config.MANDATORY_DISCLAIMER,
                equivalence_disclaimer=config.EQUIVALENCE_DISCLAIMER
            )

        # 6. Handle Mode: FULL_MULTIMODAL
        with tracker.time_block("pathology"):
            path_summary, p_repr = self.pathology_service.process_tiles(
                v_tiles, backbone=backbone, aggregation_method=agg_method
            )

        with tracker.time_block("temporal"):
            temp_summary, t_repr, t_inputs = self.temporal_service.process_observations(
                v_history, architecture=temporal_arch
            )

        with tracker.time_block("fusion"):
            fusion_summary, raw_fusion = self.fusion_service.fuse(
                pathology_repr=p_repr,
                temporal_repr=t_repr,
                fusion_mode=fusion_mode,
                p_malignant=path_summary.malignant_probability,
                p_progression=temp_summary.progression_probability,
                ctdna_forecast=temp_summary.ctdna_30d_forecast
            )

        # 7. Calibration & Epistemic Uncertainty
        with tracker.time_block("uncertainty"):
            if 'progression_logits' in raw_fusion:
                cal_prob, cal_status = self.uncertainty_service.calibrate_probability(
                    raw_fusion['progression_logits']
                )
            else:
                cal_prob = fusion_summary.progression_probability
                cal_status = "NOT_CALIBRATED"

            fusion_model = self.fusion_service.get_fusion_model(fusion_mode)
            if hasattr(fusion_model, 'shared_mlp'):
                uncertainty_out = self.uncertainty_service.estimate_uncertainty(
                    fusion_model, p_repr, t_repr, current_probability=cal_prob
                )
            else:
                uncertainty_out = schemas.UncertaintyOutput(
                    confidence=round(max(cal_prob, 1.0 - cal_prob), 4),
                    calibration_status=cal_status
                )
            uncertainty_out.calibration_status = cal_status

        # 8. Out-of-Distribution Detection
        with tracker.time_block("ood"):
            if 'fused_representation' in raw_fusion:
                ood_out = self.ood_service.evaluate_representation(raw_fusion['fused_representation'])
            else:
                ood_out = schemas.OODOutput(status="NOT_EVALUATED")

        # 9. Risk Stratification
        risk_out = RiskEngine.stratify_risk(cal_prob, uncertainty=uncertainty_out.uncertainty_score)

        total_ms = (time.perf_counter() - total_start) * 1000.0
        tracker.timings["total_ms"] = round(total_ms, 2)

        logging_utils.log_inference_event(
            patient_id=patient_id,
            inference_mode=mode,
            model_version=active_version,
            risk_level=risk_out.level,
            ood_status=ood_out.status,
            latency_ms=total_ms
        )

        return schemas.PatientInferenceResponse(
            patient_id=patient_id,
            inference_mode=mode,
            available_modalities=avail_modalities,
            model_configuration=active_version,
            progression_probability=cal_prob,
            ctdna_forecast_30d=fusion_summary.ctdna_forecast,
            risk=risk_out,
            pathology=path_summary,
            temporal=temp_summary,
            fusion=fusion_summary,
            uncertainty=uncertainty_out,
            ood=ood_out,
            latency=schemas.LatencySummary(
                validation_ms=tracker.timings.get("validation", 0.0),
                pathology_ms=tracker.timings.get("pathology", 0.0),
                temporal_ms=tracker.timings.get("temporal", 0.0),
                fusion_ms=tracker.timings.get("fusion", 0.0),
                uncertainty_ms=tracker.timings.get("uncertainty", 0.0),
                ood_ms=tracker.timings.get("ood", 0.0),
                total_ms=round(total_ms, 2)
            ),
            system={"version": active_version, "environment": "research-prototype"},
            mandatory_disclaimer=config.MANDATORY_DISCLAIMER,
            equivalence_disclaimer=config.EQUIVALENCE_DISCLAIMER
        )
