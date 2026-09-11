"""Demographic distribution calculation."""
import pandas as pd
import numpy as np
from typing import Dict, Any, List

class DemographicDistributionBuilder:
    def __init__(self, quantiles: List[float] = [0.05, 0.25, 0.50, 0.75, 0.95]):
        self.quantiles = quantiles

    def compute(self, df: pd.DataFrame, source: str = "PROJECT_STAGE1", version: str = "v1.0") -> pd.DataFrame:
        records = []
        # Age distribution
        if "age" in df.columns:
            age_s = df["age"].dropna()
            q_vals = age_s.quantile(self.quantiles).to_dict()
            records.append({
                "feature": "age",
                "type": "numerical",
                "unit": "years",
                "count": int(len(age_s)),
                "mean": float(age_s.mean()),
                "median": float(age_s.median()),
                "std": float(age_s.std()),
                "min": float(age_s.min()),
                "max": float(age_s.max()),
                "p05": float(q_vals.get(0.05, 0)),
                "p25": float(q_vals.get(0.25, 0)),
                "p50": float(q_vals.get(0.50, 0)),
                "p75": float(q_vals.get(0.75, 0)),
                "p95": float(q_vals.get(0.95, 0)),
                "missing_rate": float(df["age"].isna().mean()),
                "source": source,
                "version": version
            })

        # Sex distribution
        if "sex" in df.columns:
            counts = df["sex"].value_counts(normalize=False).to_dict()
            freqs = df["sex"].value_counts(normalize=True).to_dict()
            for cat, cnt in counts.items():
                records.append({
                    "feature": f"sex_{cat}",
                    "type": "categorical",
                    "unit": "category",
                    "count": int(cnt),
                    "mean": float(freqs[cat]),
                    "median": float(freqs[cat]),
                    "std": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "p05": 0.0, "p25": 0.0, "p50": float(freqs[cat]), "p75": 1.0, "p95": 1.0,
                    "missing_rate": float(df["sex"].isna().mean()),
                    "source": source,
                    "version": version
                })

        return pd.DataFrame(records)
