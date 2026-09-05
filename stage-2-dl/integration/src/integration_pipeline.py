"""
Stage 2 Deep Learning - Multimodal Patient-Level Integration Pipeline

Unites frozen ResNet-18 Pathology CNN and 2-layer BiLSTM Temporal Forecaster
into a single, standardized, patient-level inference service.

IMPORTANT FUSION VALIDATION NOTICE:
The default fusion weights and alert thresholds in this pipeline are prototype
engineering parameters only. They are NOT clinically validated.
They were NOT optimized or tuned using the locked test set.
LOW, MODERATE, and HIGH represent prototype engineering alert categories, NOT clinically
established risk strata.
"""
import sys
import os
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import pandas as pd
from PIL import Image

try:
    from . import config, validation, patient_aggregation, fusion
except (ImportError, ValueError):
    import config, validation, patient_aggregation, fusion

# Import frozen DL inference predictors
sys.path.insert(0, str(config.STAGE_2_DIR / 'dl'))
from src.inference import PathologyImagePredictor, TemporalBiomarkerPredictor


class MultimodalPatientPipeline:
    """
    Unified multimodal patient inference pipeline.
    Caches frozen DL models in memory for efficient, deterministic inference.
    """
    _instance = None

    def __init__(
        self,
        pathology_checkpoint: Optional[str] = None,
        temporal_checkpoint: Optional[str] = None
    ):
        pathology_chk = pathology_checkpoint or str(config.FROZEN_IMAGE_CHECKPOINT)
        temporal_chk = temporal_checkpoint or str(config.FROZEN_TEMPORAL_CHECKPOINT)

        print(f"[Pipeline Init] Loading frozen Pathology CNN from: {pathology_chk}")
        self.image_predictor = PathologyImagePredictor(checkpoint_path=pathology_chk)
        self.aggregator = patient_aggregation.PatientPathologyAggregator(predictor=self.image_predictor)

        print(f"[Pipeline Init] Loading frozen Temporal BiLSTM from: {temporal_chk}")
        self.temporal_predictor = TemporalBiomarkerPredictor(checkpoint_path=temporal_chk)

        self.pathology_checkpoint_path = pathology_chk
        self.temporal_checkpoint_path = temporal_chk
        print("[Pipeline Init] Multimodal pipeline initialized successfully.")

    @classmethod
    def get_shared_pipeline(cls) -> 'MultimodalPatientPipeline':
        """Singleton accessor for shared pipeline instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def run_patient_inference(
        self,
        patient_id: str,
        pathology_tiles: Optional[Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]]] = None,
        temporal_history: Optional[pd.DataFrame] = None,
        aggregation_method: str = config.DEFAULT_TILE_AGGREGATION,
        fusion_method: str = 'weighted_linear',
        custom_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Executes unified patient-level inference.

        Args:
            patient_id: Unique patient identifier.
            pathology_tiles: Single path/image or list of paths/images for biopsy tiles.
            temporal_history: DataFrame of historical visits (must have days_from_baseline <= 90).
            aggregation_method: 'mean', 'median', or 'max'.
            fusion_method: 'weighted_linear' or 'rule_based'.
            custom_weights: Optional dict of fusion weights summing to 1.0.

        Returns:
            Standardized patient-level multimodal dictionary.
        """
        # 1. Validate Inputs
        v_patient_id = validation.validate_patient_id(patient_id)
        v_tiles = validation.validate_pathology_tiles(pathology_tiles)
        v_history = validation.validate_temporal_history(temporal_history)

        # 2. Process Pathology Modality
        if v_tiles:
            pathology_summary = self.aggregator.aggregate_patient_tiles(
                tiles=v_tiles,
                method=aggregation_method
            )
        else:
            pathology_summary = {
                'available': False,
                'num_tiles_analyzed': 0,
                'aggregation_method': aggregation_method,
                'malignant_probability': None,
                'benign_probability': None,
                'inflammation_probability': None,
                'primary_patient_class': None,
                'tile_predictions_breakdown': {'benign': 0, 'malignant': 0, 'inflammation': 0},
                'tile_details': [],
                'status': 'NOT_PROVIDED'
            }

        # 3. Process Temporal Modality
        if v_history is not None and len(v_history) > 0:
            temp_raw = self.temporal_predictor.predict(v_history)
            p_prog = temp_raw['predicted_progression_risk']
            vaf_30d = temp_raw['predicted_ctDNA_30d_vaf']
            vaf_norm = fusion.normalize_ctdna_risk(vaf_30d)

            temporal_summary = {
                'available': True,
                'input_sequence_length': temp_raw['input_sequence_length'],
                'max_days_from_baseline': temp_raw['max_days_from_baseline'],
                'progression_probability': p_prog,
                'predicted_progression': temp_raw['predicted_progression'],
                'predicted_ctDNA_30d_vaf': vaf_30d,
                'normalized_ctdna_risk': vaf_norm,
                'status': 'SUCCESS'
            }
        else:
            temporal_summary = {
                'available': False,
                'input_sequence_length': 0,
                'max_days_from_baseline': None,
                'progression_probability': None,
                'predicted_progression': None,
                'predicted_ctDNA_30d_vaf': None,
                'normalized_ctdna_risk': None,
                'status': 'NOT_PROVIDED'
            }

        # 4. Multimodal Fusion
        fusion_result = fusion.execute_multimodal_fusion(
            pathology_summary=pathology_summary,
            temporal_summary=temporal_summary,
            fusion_method=fusion_method,
            custom_weights=custom_weights
        )

        # 5. Compile Standardized Result
        timestamp = datetime.now(timezone.utc).isoformat()

        output = {
            'patient_id': v_patient_id,
            'modality_availability': {
                'pathology': pathology_summary['available'],
                'temporal': temporal_summary['available']
            },
            'modality_status': fusion_result['modality_status'],
            'pathology_summary': pathology_summary,
            'temporal_summary': temporal_summary,
            'multimodal_fusion': {
                'fusion_method': fusion_result['fusion_method'],
                'weights_used': fusion_result['weights_used'],
                'prototype_multimodal_risk_score': fusion_result['multimodal_risk_score'],
                'prototype_alert_level': fusion_result['prototype_alert_level'],
                'normalized_ctdna_risk': fusion_result['normalized_ctdna_risk'],
                'score_components': fusion_result['score_components']
            },
            'engineering_explanation': fusion_result['engineering_explanation'],
            'provenance': {
                'timestamp': timestamp,
                'data_source': config.DATA_SOURCE,
                'clinical_validation_status': config.CLINICAL_VALIDATION_STATUS,
                'pathology_model_checkpoint': str(self.pathology_checkpoint_path),
                'temporal_model_checkpoint': str(self.temporal_checkpoint_path),
                'mandatory_disclaimer': config.MANDATORY_DISCLAIMER,
                'equivalence_disclaimer': config.EQUIVALENCE_DISCLAIMER
            }
        }

        return output


def run_patient_inference(
    patient_id: str,
    pathology_tiles: Optional[Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]]] = None,
    temporal_history: Optional[pd.DataFrame] = None,
    aggregation_method: str = config.DEFAULT_TILE_AGGREGATION,
    fusion_method: str = 'weighted_linear',
    custom_weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Convenience functional wrapper around MultimodalPatientPipeline."""
    pipeline = MultimodalPatientPipeline.get_shared_pipeline()
    return pipeline.run_patient_inference(
        patient_id=patient_id,
        pathology_tiles=pathology_tiles,
        temporal_history=temporal_history,
        aggregation_method=aggregation_method,
        fusion_method=fusion_method,
        custom_weights=custom_weights
    )


def _load_sample_patient_data(patient_id: str):
    """Loads sample data for a patient from the development/validation sets."""
    img_df = pd.read_csv(config.IMAGE_METADATA_PATH)
    bio_df = pd.read_csv(config.BIOMARKERS_PATH)

    pt_imgs = img_df[img_df['patient_id'] == patient_id]
    tile_paths = []
    for _, row in pt_imgs.iterrows():
        p = row['image_path']
        if not os.path.exists(p):
            rel = row.get('relative_path', '')
            alt = config.STAGE_2_DIR / rel.replace('data-engineering/data-engineering/', 'data-engineering/')
            p = str(alt)
        if os.path.exists(p):
            tile_paths.append(p)

    pt_bio = bio_df[
        (bio_df['patient_id'] == patient_id) &
        (bio_df['is_input_window'] == 1) &
        (bio_df['days_from_baseline'] <= config.FORECAST_SPLIT_DAY)
    ].copy()

    # Drop target columns for clean inference input
    for col in config.FORBIDDEN_COLUMNS:
        if col in pt_bio.columns:
            pt_bio = pt_bio.drop(columns=[col])

    return tile_paths, pt_bio


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multimodal Patient Inference CLI")
    parser.add_argument('--patient-id', type=str, default='PAT-0001', help="Patient ID to evaluate")
    parser.add_argument('--aggregation', type=str, default='mean', choices=['mean', 'median', 'max'])
    parser.add_argument('--fusion', type=str, default='weighted_linear', choices=['weighted_linear', 'rule_based'])
    args = parser.parse_args()

    print(f"\n--- Running Multimodal Inference for {args.patient_id} ---")
    tiles, bio_df = _load_sample_patient_data(args.patient_id)
    print(f"Found {len(tiles)} tiles and {len(bio_df)} historical visits for {args.patient_id}.")

    res = run_patient_inference(
        patient_id=args.patient_id,
        pathology_tiles=tiles,
        temporal_history=bio_df,
        aggregation_method=args.aggregation,
        fusion_method=args.fusion
    )

    import json
    # Print pretty JSON excluding large tile details
    res_display = dict(res)
    if 'tile_details' in res_display['pathology_summary']:
        res_display['pathology_summary'] = dict(res_display['pathology_summary'])
        res_display['pathology_summary']['tile_details'] = f"[{len(tiles)} tile details omitted from stdout]"

    print("\n--- Standardized Multimodal Result ---")
    print(json.dumps(res_display, indent=2))
