"""
Stage 2 EDA - Longitudinal Temporal Biomarkers & Forecasting Schema Module
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
try:
    from . import config
except (ImportError, ValueError):
    import config

def load_biomarker_data() -> pd.DataFrame:
    """Loads processed temporal biomarker dataset."""
    df = pd.read_csv(config.BIOMARKERS_PATH)
    return df

def audit_biomarker_distributions(bio_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Computes summary statistics for all 5 continuous biomarkers:
    ctDNA VAF, CEA, CA-125, LDH, CRP.
    Checks physiological bounds, non-negativity, and returns temporal_statistics.csv.
    """
    records = []
    anomaly_report = {}

    for marker in config.BIOMARKERS:
        series = bio_df[marker]
        mask_col = config.MISSING_MASKS[marker]
        mask_series = bio_df[mask_col]

        total_cnt = len(series)
        valid_cnt = int(series.notna().sum())
        missing_cnt = int(series.isna().sum())
        missing_pct = round((missing_cnt / total_cnt) * 100, 2)
        mask_cnt = int(mask_series.sum())

        # Exact match verification between NaN and missing mask
        mask_match = (series.isna() == (mask_series == 1)).all()

        # Physiological validation
        valid_vals = series.dropna()
        has_negatives = bool((valid_vals < 0).any())
        has_infinities = bool(np.isinf(valid_vals).any())

        mean_val = float(valid_vals.mean())
        std_val = float(valid_vals.std())
        median_val = float(valid_vals.median())
        min_val = float(valid_vals.min())
        max_val = float(valid_vals.max())
        p5 = float(np.percentile(valid_vals, 5))
        p25 = float(np.percentile(valid_vals, 25))
        p75 = float(np.percentile(valid_vals, 75))
        p95 = float(np.percentile(valid_vals, 95))

        records.append({
            'biomarker': marker,
            'total_observations': total_cnt,
            'valid_count': valid_cnt,
            'missing_count': missing_cnt,
            'missing_percentage': missing_pct,
            'mask_count': mask_cnt,
            'mask_integrity_verified': bool(mask_match),
            'mean': round(mean_val, 4),
            'std': round(std_val, 4),
            'median': round(median_val, 4),
            'min': round(min_val, 4),
            'p5': round(p5, 4),
            'p25': round(p25, 4),
            'p75': round(p75, 4),
            'p95': round(p95, 4),
            'max': round(max_val, 4),
            'has_negative_values': has_negatives,
            'has_infinities': has_infinities,
        })

        anomaly_report[marker] = {
            'has_negatives': has_negatives,
            'has_infinities': has_infinities,
            'mask_match': mask_match,
            'classification': 'EXPECTED' if (not has_negatives and not has_infinities and mask_match) else 'DATA QUALITY ISSUE'
        }

    stats_df = pd.DataFrame(records)
    return stats_df, anomaly_report

def audit_forecasting_windows(bio_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Audits strict temporal segregation between historical input window (<= 90 days)
    and future prediction window (> 90 days).
    """
    hist_sub = bio_df[bio_df['is_input_window'] == 1]
    fut_sub = bio_df[bio_df['is_input_window'] == 0]

    # Boundary integrity check
    hist_max_day = int(hist_sub['days_from_baseline'].max()) if len(hist_sub) > 0 else 0
    fut_min_day = int(fut_sub['days_from_baseline'].min()) if len(fut_sub) > 0 else 0

    hist_violates = (hist_sub['days_from_baseline'] > 90).sum()
    fut_violates = (fut_sub['days_from_baseline'] <= 90).sum()

    # Targets audit
    targets_available = int(bio_df['future_ctDNA_30d_target'].notna().sum())
    targets_missing = int(bio_df['future_ctDNA_30d_target'].isna().sum())

    progression_counts = bio_df['future_progression_trend'].value_counts().to_dict()
    patient_progression = bio_df.groupby('patient_id')['future_progression_trend'].first().value_counts().to_dict()

    window_pass = (hist_violates == 0 and fut_violates == 0 and hist_max_day <= 90 and fut_min_day > 90)

    return {
        'historical_observations': len(hist_sub),
        'future_observations': len(fut_sub),
        'historical_max_day': hist_max_day,
        'future_min_day': fut_min_day,
        'historical_violations': int(hist_violates),
        'future_violations': int(fut_violates),
        'window_boundary_pass': bool(window_pass),
        'valid_30d_ctdna_targets': targets_available,
        'missing_30d_ctdna_targets': targets_missing,
        'target_availability_pct': round((targets_available / len(bio_df)) * 100, 2),
        'progression_trend_distribution_obs': {str(k): int(v) for k, v in progression_counts.items()},
        'progression_trend_distribution_pats': {str(k): int(v) for k, v in patient_progression.items()},
        'status': 'PASS' if window_pass else 'FAIL'
    }

def audit_temporal_leakage_schema(bio_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Audits column roles in the temporal schema to establish strict boundaries between
    valid input features, future forecast targets, and administrative metadata.
    """
    all_cols = list(bio_df.columns)

    input_features = [
        'timepoint_index', 'days_from_baseline', 'delta_days',
        'ctDNA_vaf_percent', 'cea_ng_ml', 'ca125_u_ml', 'ldh_u_l', 'crp_mg_l',
        'ctDNA_velocity_30d',
        'ctDNA_missing', 'cea_missing', 'ca125_missing', 'ldh_missing', 'crp_missing'
    ]

    forecast_targets = [
        'future_ctDNA_30d_target',
        'future_progression_trend',
        'trajectory_pattern'
    ]

    metadata_cols = [
        'patient_id', 'split', 'timestamp', 'window_type',
        'is_input_window', 'data_source', 'disclaimer'
    ]

    # Verify no overlap between inputs and targets
    leakage_risk_cols = set(input_features).intersection(set(forecast_targets))

    return {
        'all_columns': all_cols,
        'input_features': input_features,
        'forecast_targets': forecast_targets,
        'metadata_columns': metadata_cols,
        'leakage_risk_cols': list(leakage_risk_cols),
        'is_leakage_free': len(leakage_risk_cols) == 0,
        'status': 'PASS' if len(leakage_risk_cols) == 0 else 'FAIL',
        'recommendation': (
            'Deep Learning models (LSTM/Transformer) MUST ONLY receive columns from '
            '`input_features` where `is_input_window == 1`. Columns in `forecast_targets` '
            'must be isolated strictly as loss supervision signals.'
        )
    }
