"""
Feature Diagnostics and Stability Metrics Module for Stage 3 Clinical NLP.
Evaluates dimensions, missingness, infinite values, zero-variance columns,
distribution shifts, and schema consistency across Train, Val, and Locked Test sets.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import ks_2samp


def evaluate_feature_matrix(
    X_matrix: csr_matrix,
    struct_df: pd.DataFrame,
    numeric_cols: List[str],
    split_name: str = "validation"
) -> Dict[str, Any]:
    """
    Evaluate technical integrity of the 1,012-dimensional feature matrix and structured dataframe.
    """
    n_samples, n_features = X_matrix.shape

    # Check for NaN / Inf in sparse matrix data
    data_arr = X_matrix.data
    nan_count = int(np.isnan(data_arr).sum()) if len(data_arr) > 0 else 0
    inf_count = int(np.isinf(data_arr).sum()) if len(data_arr) > 0 else 0

    # Calculate column variances to detect constant / dead features
    mean_dense = np.array(X_matrix.mean(axis=0)).flatten()
    sq_matrix = X_matrix.copy()
    sq_matrix.data = sq_matrix.data ** 2
    mean_sq = np.array(sq_matrix.mean(axis=0)).flatten()
    variances = mean_sq - (mean_dense ** 2)
    zero_variance_cols = int(np.sum(variances <= 1e-12))

    # Structured features statistics
    struct_stats = {}
    for col in numeric_cols:
        if col in struct_df.columns:
            vals = struct_df[col].astype(float).values
            struct_stats[col] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
                "null_count": int(struct_df[col].isnull().sum())
            }

    return {
        "split_name": split_name,
        "n_samples": n_samples,
        "total_features": n_features,
        "expected_features": 1012,
        "is_dimension_valid": (n_features == 1012),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "zero_variance_features_count": zero_variance_cols,
        "zero_variance_ratio": round(zero_variance_cols / n_features, 4),
        "structured_feature_statistics": struct_stats
    }


def compute_feature_drift(
    train_struct_df: pd.DataFrame,
    eval_struct_df: pd.DataFrame,
    numeric_cols: List[str]
) -> Dict[str, Any]:
    """
    Perform Kolmogorov-Smirnov (KS) two-sample test to detect distribution drift
    in structured concept features between Train and Validation/Test.
    """
    drift_results = {}
    drift_detected_count = 0

    for col in numeric_cols:
        if col in train_struct_df.columns and col in eval_struct_df.columns:
            tr_vals = train_struct_df[col].values
            ev_vals = eval_struct_df[col].values
            ks_stat, p_val = ks_2samp(tr_vals, ev_vals)
            # Drift flagged if p-value < 0.01 and ks_stat > 0.08
            is_drift = (p_val < 0.01 and ks_stat > 0.08)
            if is_drift:
                drift_detected_count += 1

            drift_results[col] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_val), 6),
                "drift_flag": bool(is_drift)
            }

    return {
        "total_features_tested": len(numeric_cols),
        "drift_detected_features_count": drift_detected_count,
        "feature_drift_details": drift_results
    }
