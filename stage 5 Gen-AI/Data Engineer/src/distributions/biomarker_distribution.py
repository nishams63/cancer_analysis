"""Biomarker distribution compilation."""
import pandas as pd
import numpy as np
from typing import List

class BiomarkerDistributionBuilder:
    def __init__(self, quantiles: List[float] = [0.05, 0.25, 0.50, 0.75, 0.95]):
        self.quantiles = quantiles

    def compute(self, df: pd.DataFrame, source: str = "PROJECT_STAGE1", version: str = "v1.0") -> pd.DataFrame:
        biomarker_cols = {
            "ctdna_level": "ng/mL",
            "tumor_marker_level": "U/mL",
            "inflammation_marker": "mg/L",
            "gene_expression_score": "score",
            "hemoglobin": "g/dL",
            "white_blood_cell_count": "10^3/uL",
            "platelet_count": "10^3/uL",
            "creatinine_level": "mg/dL",
            "liver_function_marker": "U/L",
            "systolic_bp": "mmHg",
            "diastolic_bp": "mmHg",
            "heart_rate": "bpm",
            "oxygen_saturation": "%"
        }
        records = []
        for col, unit in biomarker_cols.items():
            if col in df.columns:
                s = df[col].dropna()
                if len(s) == 0:
                    continue
                q = s.quantile(self.quantiles).to_dict()
                records.append({
                    "biomarker_name": col,
                    "unit": unit,
                    "count": int(len(s)),
                    "mean": round(float(s.mean()), 4),
                    "median": round(float(s.median()), 4),
                    "std": round(float(s.std()), 4),
                    "min": round(float(s.min()), 4),
                    "max": round(float(s.max()), 4),
                    "p05": round(float(q.get(0.05, 0)), 4),
                    "p25": round(float(q.get(0.25, 0)), 4),
                    "p50": round(float(q.get(0.50, 0)), 4),
                    "p75": round(float(q.get(0.75, 0)), 4),
                    "p95": round(float(q.get(0.95, 0)), 4),
                    "missing_rate": round(float(df[col].isna().mean()), 4),
                    "source": source,
                    "version": version
                })
        return pd.DataFrame(records)
