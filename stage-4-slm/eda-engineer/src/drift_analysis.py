"""
Temporal and Distribution Drift Analysis Module for Stage 4 EDA.
Audits document timestamps and evaluates statistical drift between Train,
Validation, and Test splits using Kolmogorov-Smirnov and Jensen-Shannon tests.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, chisquare


class DriftAnalyzer:
    """Evaluates temporal distributions and cross-split statistical drift."""

    def analyze_temporal_distribution(
        self, df: pd.DataFrame, stage3_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Audits date distributions across Train, Validation, and Test partitions.
        Extracts document_date from Stage 3 reference or checks if present in df.
        """
        if "document_date" in df.columns:
            date_col = df["document_date"]
        elif stage3_df is not None and "document_date" in stage3_df.columns:
            date_map = dict(zip(stage3_df["document_id"], stage3_df["document_date"]))
            date_col = df["note_id"].map(date_map)
        else:
            return {"temporal_analysis_available": False, "reason": "No document_date field located"}

        df_temp = df.copy()
        df_temp["doc_date"] = pd.to_datetime(date_col, errors="coerce")

        temporal_by_split = {}
        for s in df["split"].unique():
            subset = df_temp[df_temp["split"] == s]["doc_date"].dropna()
            if len(subset) > 0:
                temporal_by_split[s] = {
                    "min_date": str(subset.min().date()),
                    "max_date": str(subset.max().date()),
                    "median_date": str(subset.median().date()),
                    "record_count": len(subset)
                }

        # Check if splits follow strict chronological progression (Train < Val < Test)
        is_chronological = False
        if "TRAIN" in temporal_by_split and "VALIDATION" in temporal_by_split and "TEST" in temporal_by_split:
            train_max = temporal_by_split["TRAIN"]["max_date"]
            val_min = temporal_by_split["VALIDATION"]["min_date"]
            val_max = temporal_by_split["VALIDATION"]["max_date"]
            test_min = temporal_by_split["TEST"]["min_date"]
            if train_max <= val_min and val_max <= test_min:
                is_chronological = True

        return {
            "temporal_analysis_available": True,
            "is_strictly_chronological": is_chronological,
            "splits": temporal_by_split,
            "design_interpretation": (
                "Patient-stratified cohorts spanning full temporal window rather than retrospective/prospective time split."
                if not is_chronological
                else "Strict chronological split."
            )
        }

    def analyze_distribution_drift(
        self,
        df: pd.DataFrame,
        source_tokens: np.ndarray,
        risk_series: pd.Series,
        entity_densities: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculates cross-split distribution drift between Train vs Validation and Train vs Test:
        - Continuous distributions (token lengths, entity densities): Two-Sample Kolmogorov-Smirnov (KS) test.
        - Categorical distributions (risk classes): Jensen-Shannon divergence.
        """
        df_eval = pd.DataFrame({
            "split": df["split"].values,
            "source_tokens": source_tokens,
            "risk_tier": risk_series.values,
            "entity_density": entity_densities
        })

        train_sub = df_eval[df_eval["split"] == "TRAIN"]
        val_sub = df_eval[df_eval["split"] == "VALIDATION"]
        test_sub = df_eval[df_eval["split"] == "TEST"]

        def run_ks(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
            if len(a) == 0 or len(b) == 0:
                return {"ks_statistic": 0.0, "p_value": 1.0}
            stat, pval = ks_2samp(a, b)
            return {"ks_statistic": float(round(float(stat), 4)), "p_value": float(round(float(pval), 4))}

        def run_js(cat_a: pd.Series, cat_b: pd.Series, categories: List[str]) -> float:
            counts_a = cat_a.value_counts()
            counts_b = cat_b.value_counts()
            prob_a = [counts_a.get(c, 0) / len(cat_a) for c in categories] if len(cat_a) > 0 else [0.33, 0.33, 0.33]
            prob_b = [counts_b.get(c, 0) / len(cat_b) for c in categories] if len(cat_b) > 0 else [0.33, 0.33, 0.33]
            return float(round(float(jensenshannon(prob_a, prob_b, base=2)), 4))

        risk_tiers = ["Low", "Moderate", "High"]

        drift_results = {}
        for target_name, target_sub in [("VALIDATION", val_sub), ("TEST", test_sub)]:
            if len(target_sub) == 0:
                continue

            ks_tokens = run_ks(train_sub["source_tokens"].values, target_sub["source_tokens"].values)
            ks_density = run_ks(train_sub["entity_density"].values, target_sub["entity_density"].values)
            js_risk = run_js(train_sub["risk_tier"], target_sub["risk_tier"], risk_tiers)

            # Flag drift if KS statistic > 0.10 and p-value < 0.05, or JS divergence > 0.15
            drift_flag = (ks_tokens["ks_statistic"] > 0.10 and ks_tokens["p_value"] < 0.05) or (js_risk > 0.15)

            drift_results[f"TRAIN_vs_{target_name}"] = {
                "token_length_ks_test": ks_tokens,
                "entity_density_ks_test": ks_density,
                "risk_class_js_divergence": js_risk,
                "statistical_drift_detected": drift_flag
            }

        return drift_results
