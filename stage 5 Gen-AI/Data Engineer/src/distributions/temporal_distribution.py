"""Longitudinal and temporal pattern distribution calculation."""
import pandas as pd

class TemporalDistributionBuilder:
    def compute(self, df_longitudinal: pd.DataFrame, source: str = "PROJECT_STAGE2", version: str = "v1.0") -> pd.DataFrame:
        records = []
        if "delta_days" in df_longitudinal.columns:
            d_days = df_longitudinal["delta_days"].dropna()
            records.append({
                "transition_type": "treatment_to_followup_interval",
                "count": int(len(d_days)),
                "mean_days": round(float(d_days.mean()), 2),
                "median_days": round(float(d_days.median()), 2),
                "min_days": round(float(d_days.min()), 2),
                "max_days": round(float(d_days.max()), 2),
                "std_days": round(float(d_days.std()), 2),
                "source": source,
                "version": version
            })
        if "days_from_baseline" in df_longitudinal.columns:
            b_days = df_longitudinal["days_from_baseline"].dropna()
            records.append({
                "transition_type": "baseline_to_observation_span",
                "count": int(len(b_days)),
                "mean_days": round(float(b_days.mean()), 2),
                "median_days": round(float(b_days.median()), 2),
                "min_days": round(float(b_days.min()), 2),
                "max_days": round(float(b_days.max()), 2),
                "std_days": round(float(b_days.std()), 2),
                "source": source,
                "version": version
            })
        return pd.DataFrame(records)
