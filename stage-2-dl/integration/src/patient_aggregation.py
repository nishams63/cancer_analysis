"""
Stage 2 Deep Learning - Patient-Level Pathology Tile Aggregator Module

Aggregates predictions across multiple whole-slide biopsy tiles for a single patient.
Supports:
  - mean malignant probability (default baseline)
  - median malignant probability
  - maximum malignant probability
Tracks tile-level variance, dispersion, and class counts for full auditability.
"""
import sys
from pathlib import Path
from typing import List, Dict, Any, Union
import numpy as np
from PIL import Image

try:
    from . import config, validation
except (ImportError, ValueError):
    import config, validation

# Import frozen DL inference predictor
sys.path.insert(0, str(config.STAGE_2_DIR / 'dl'))
from src.inference import PathologyImagePredictor


class PatientPathologyAggregator:
    """
    Patient-level aggregator for multiple histopathology tiles.
    Wraps the frozen PathologyImagePredictor.
    """
    def __init__(self, predictor: PathologyImagePredictor = None, checkpoint_path: str = None):
        if predictor is not None:
            self.predictor = predictor
        else:
            chk = checkpoint_path or str(config.FROZEN_IMAGE_CHECKPOINT)
            self.predictor = PathologyImagePredictor(checkpoint_path=chk)

    def aggregate_patient_tiles(
        self,
        tiles: List[Union[str, Path, Image.Image]],
        method: str = config.DEFAULT_TILE_AGGREGATION
    ) -> Dict[str, Any]:
        """
        Runs inference on all patient tiles and computes patient-level aggregate statistics.

        Args:
            tiles: List of image paths or PIL Image objects.
            method: Aggregation strategy ('mean', 'median', 'max').

        Returns:
            Dictionary containing patient-level probabilities, breakdown, and audit metadata.
        """
        agg_method = validation.validate_aggregation_method(method)

        if not tiles:
            return {
                'available': False,
                'num_tiles_analyzed': 0,
                'aggregation_method': agg_method,
                'malignant_probability': None,
                'benign_probability': None,
                'inflammation_probability': None,
                'primary_patient_class': None,
                'tile_predictions_breakdown': {'benign': 0, 'malignant': 0, 'inflammation': 0},
                'tile_details': [],
                'status': 'NO_PATHOLOGY_TILES'
            }

        tile_results = []
        p_benign_list = []
        p_malignant_list = []
        p_inflammation_list = []
        pred_counts = {'benign': 0, 'malignant': 0, 'inflammation': 0}

        for idx, tile_item in enumerate(tiles):
            pred_res = self.predictor.predict(tile_item)
            probs = pred_res['probabilities']
            pred_class = pred_res['prediction']

            p_benign_list.append(probs['benign'])
            p_malignant_list.append(probs['malignant'])
            p_inflammation_list.append(probs['inflammation'])
            pred_counts[pred_class] = pred_counts.get(pred_class, 0) + 1

            tile_identifier = str(tile_item) if isinstance(tile_item, (str, Path)) else f"tile_{idx+1}"
            tile_results.append({
                'tile_index': idx + 1,
                'tile_identifier': tile_identifier,
                'prediction': pred_class,
                'confidence': pred_res['confidence'],
                'probabilities': probs
            })

        # Apply aggregation method
        if agg_method == 'mean':
            p_mal = float(np.mean(p_malignant_list))
            p_ben = float(np.mean(p_benign_list))
            p_inf = float(np.mean(p_inflammation_list))
        elif agg_method == 'median':
            p_mal = float(np.median(p_malignant_list))
            p_ben = float(np.median(p_benign_list))
            p_inf = float(np.median(p_inflammation_list))
        elif agg_method == 'max':
            p_mal = float(np.max(p_malignant_list))
            # Normalize other classes proportionally to keep sum roughly 1.0
            p_ben = float(np.mean(p_benign_list))
            p_inf = float(np.mean(p_inflammation_list))

        # Re-normalize to ensure sum to 1.0
        total_p = p_mal + p_ben + p_inf
        if total_p > 0:
            p_mal_norm = p_mal / total_p
            p_ben_norm = p_ben / total_p
            p_inf_norm = p_inf / total_p
        else:
            p_mal_norm, p_ben_norm, p_inf_norm = 0.0, 0.0, 0.0

        # Determine dominant patient-level morphological class
        class_candidates = [
            ('benign', p_ben_norm),
            ('malignant', p_mal_norm),
            ('inflammation', p_inf_norm)
        ]
        dominant_class = max(class_candidates, key=lambda x: x[1])[0]

        # Calculate tile dispersion / consensus metrics
        mal_std = float(np.std(p_malignant_list)) if len(p_malignant_list) > 1 else 0.0
        mal_min = float(np.min(p_malignant_list))
        mal_max = float(np.max(p_malignant_list))

        return {
            'available': True,
            'num_tiles_analyzed': len(tiles),
            'aggregation_method': agg_method,
            'malignant_probability': round(p_mal_norm, 4),
            'benign_probability': round(p_ben_norm, 4),
            'inflammation_probability': round(p_inf_norm, 4),
            'primary_patient_class': dominant_class,
            'tile_predictions_breakdown': pred_counts,
            'malignant_dispersion': {
                'std': round(mal_std, 4),
                'min': round(mal_min, 4),
                'max': round(mal_max, 4)
            },
            'tile_details': tile_results,
            'status': 'SUCCESS'
        }
