"""
Stage 2 Deep Learning - Input Integrity & Anti-Leakage Validation Module

Strictly enforces:
1. Temporal historical boundary: days_from_baseline <= 90
2. Anti-leakage target exclusion: forbids future_ctDNA_30d_target & future_progression_trend
3. Modality structure and physical tile existence
4. Fusion parameter bounds
"""
import os
from pathlib import Path
from typing import List, Union, Optional
import pandas as pd
from PIL import Image

try:
    from . import config
except (ImportError, ValueError):
    import config


class ValidationError(ValueError):
    """Raised when an input violates data integrity or anti-leakage constraints."""
    pass


def validate_patient_id(patient_id: str) -> str:
    """Validates patient identifier."""
    if not isinstance(patient_id, str) or not patient_id.strip():
        raise ValidationError("Patient ID must be a non-empty string.")
    return patient_id.strip()


def validate_temporal_history(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Validates patient temporal DataFrame against anti-leakage and schema rules.
    Strictly checks that days_from_baseline <= 90 and targets are absent.
    """
    if df is None:
        return None

    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"Temporal history must be a pandas DataFrame, got {type(df)}.")

    if len(df) == 0:
        return None

    # Check for forbidden future targets
    leaked_cols = [col for col in config.FORBIDDEN_COLUMNS if col in df.columns]
    if leaked_cols:
        raise ValidationError(
            f"Anti-leakage violation: Forbidden target column(s) detected in temporal input: {leaked_cols}. "
            "Future targets must NEVER be passed to the inference pipeline."
        )

    # Check for required time column
    if 'days_from_baseline' not in df.columns:
        raise ValidationError("Temporal history DataFrame must contain 'days_from_baseline' column.")

    # Strict historical window boundary check: days <= 90
    max_day = df['days_from_baseline'].max()
    if max_day > config.FORECAST_SPLIT_DAY:
        violating_count = (df['days_from_baseline'] > config.FORECAST_SPLIT_DAY).sum()
        raise ValidationError(
            f"Forecasting boundary violation: Found {violating_count} observations exceeding Day {config.FORECAST_SPLIT_DAY} "
            f"(maximum observed day: {max_day}). Model only accepts historical induction window data (days <= 90)."
        )

    # Check for at least one numerical biomarker column
    num_cols = ['ctDNA_vaf_percent', 'cea_ng_ml', 'ca125_u_ml', 'ldh_u_l', 'crp_mg_l']
    present_cols = [c for c in num_cols if c in df.columns]
    if not present_cols:
        raise ValidationError(f"Temporal history must contain at least one biomarker column among: {num_cols}")

    return df.copy().sort_values('days_from_baseline').reset_index(drop=True)


def validate_pathology_tiles(tiles_input: Union[None, str, Path, Image.Image, List[Union[str, Path, Image.Image]]]) -> List[Union[str, Path, Image.Image]]:
    """
    Validates image tiles input. Supports single tile, list of tiles, or file paths.
    """
    if tiles_input is None:
        return []

    if isinstance(tiles_input, (str, Path, Image.Image)):
        tiles_list = [tiles_input]
    elif isinstance(tiles_input, list):
        tiles_list = tiles_input
    else:
        raise ValidationError(f"Unsupported tiles input type: {type(tiles_input)}")

    validated_tiles = []
    for idx, item in enumerate(tiles_list):
        if isinstance(item, (str, Path)):
            path_str = str(item)
            if not os.path.exists(path_str):
                raise ValidationError(f"Pathology tile file not found: {path_str}")
            try:
                with Image.open(path_str) as img:
                    img.verify()
                validated_tiles.append(path_str)
            except Exception as e:
                raise ValidationError(f"Corrupt or unreadable image file at {path_str}: {e}")
        elif isinstance(item, Image.Image):
            validated_tiles.append(item)
        else:
            raise ValidationError(f"Tile item at index {idx} has invalid type: {type(item)}")

    return validated_tiles


def validate_fusion_weights(weights: dict) -> dict:
    """Validates that fusion weights are non-negative and sum to 1.0."""
    required_keys = {'pathology_malignant', 'temporal_progression', 'ctdna_vaf_risk'}
    if not required_keys.issubset(weights.keys()):
        raise ValidationError(f"Fusion weights dict must contain keys: {required_keys}")

    for k, v in weights.items():
        if not isinstance(v, (int, float)) or v < 0:
            raise ValidationError(f"Fusion weight for '{k}' must be a non-negative float, got {v}")

    total = sum(weights[k] for k in required_keys)
    if abs(total - 1.0) > 1e-4:
        raise ValidationError(f"Fusion weights must sum to 1.0, got {total:.4f}")

    return weights


def validate_aggregation_method(method: str) -> str:
    """Validates tile aggregation strategy."""
    method_lower = method.lower()
    if method_lower not in config.ALLOWED_TILE_AGGREGATIONS:
        raise ValidationError(
            f"Invalid tile aggregation method: '{method}'. Must be one of: {config.ALLOWED_TILE_AGGREGATIONS}"
        )
    return method_lower
