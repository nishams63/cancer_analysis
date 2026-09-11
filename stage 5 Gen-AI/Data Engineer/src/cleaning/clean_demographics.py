"""Demographics cleaning and range validation."""
from typing import Dict, Any, List, Tuple
import pandas as pd

class DemographicsCleaner:
    def __init__(self, age_min: int = 18, age_max: int = 105):
        self.age_min = age_min
        self.age_max = age_max

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        cleaned_df = df.copy()
        issues = []

        # Age validation
        if "age" in cleaned_df.columns:
            for idx, val in cleaned_df["age"].items():
                row_id = cleaned_df.loc[idx, "patient_id"] if "patient_id" in cleaned_df.columns else str(idx)
                if pd.isna(val):
                    issues.append({
                        "row_id": row_id,
                        "field": "age",
                        "issue_type": "missing_value",
                        "original_value": None,
                        "action_taken": "flagged_as_missing"
                    })
                elif val < self.age_min or val > self.age_max:
                    issues.append({
                        "row_id": row_id,
                        "field": "age",
                        "issue_type": "invalid_range",
                        "original_value": val,
                        "action_taken": "quarantined_to_boundary"
                    })
                    cleaned_df.loc[idx, "age"] = max(self.age_min, min(self.age_max, val))

        # Sex standardization
        if "sex" in cleaned_df.columns:
            allowed_sex = {"Male", "Female", "Unknown"}
            for idx, val in cleaned_df["sex"].items():
                row_id = cleaned_df.loc[idx, "patient_id"] if "patient_id" in cleaned_df.columns else str(idx)
                if pd.isna(val):
                    issues.append({
                        "row_id": row_id,
                        "field": "sex",
                        "issue_type": "missing_value",
                        "original_value": None,
                        "action_taken": "imputed_unknown"
                    })
                    cleaned_df.loc[idx, "sex"] = "Unknown"
                elif str(val).strip().capitalize() not in allowed_sex:
                    issues.append({
                        "row_id": row_id,
                        "field": "sex",
                        "issue_type": "malformed_category",
                        "original_value": val,
                        "action_taken": "normalized_unknown"
                    })
                    cleaned_df.loc[idx, "sex"] = "Unknown"
                else:
                    cleaned_df.loc[idx, "sex"] = str(val).strip().capitalize()

        return cleaned_df, issues
