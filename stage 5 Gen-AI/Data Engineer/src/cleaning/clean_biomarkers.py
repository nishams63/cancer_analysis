"""Biomarker and vital sign validation against physiological bounds."""
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

class BiomarkersCleaner:
    PHYSIOLOGICAL_BOUNDS = {
        "ctdna_level": (0.0, 50.0),
        "ctDNA_vaf_percent": (0.0, 100.0),
        "tumor_marker_level": (0.0, 2000.0),
        "inflammation_marker": (0.0, 300.0),
        "gene_expression_score": (0.0, 200.0),
        "heart_rate": (30.0, 220.0),
        "systolic_bp": (60.0, 240.0),
        "diastolic_bp": (30.0, 140.0),
        "oxygen_saturation": (60.0, 100.0),
        "hemoglobin": (3.0, 25.0),
        "white_blood_cell_count": (0.1, 80.0),
        "platelet_count": (5.0, 1500.0),
        "creatinine_level": (0.1, 15.0),
        "liver_function_marker": (1.0, 1000.0)
    }

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        cleaned_df = df.copy()
        issues = []

        for col, (b_min, b_max) in self.PHYSIOLOGICAL_BOUNDS.items():
            if col in cleaned_df.columns:
                for idx, val in cleaned_df[col].items():
                    row_id = cleaned_df.loc[idx, "patient_id"] if "patient_id" in cleaned_df.columns else str(idx)
                    if pd.isna(val):
                        continue
                    if val < b_min or val > b_max:
                        issues.append({
                            "row_id": row_id,
                            "field": col,
                            "issue_type": "out_of_physiological_bounds",
                            "original_value": val,
                            "action_taken": f"clamped_to_[{b_min},{b_max}]"
                        })
                        cleaned_df.loc[idx, col] = np.clip(val, b_min, b_max)

        return cleaned_df, issues
