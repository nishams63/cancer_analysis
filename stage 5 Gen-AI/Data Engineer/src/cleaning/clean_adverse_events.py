"""Adverse events and toxicity grade cleaning."""
from typing import Dict, Any, List, Tuple
import pandas as pd

class AdverseEventCleaner:
    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        cleaned_df = df.copy()
        issues = []

        if "previous_toxicity_grade" in cleaned_df.columns:
            for idx, val in cleaned_df["previous_toxicity_grade"].items():
                row_id = cleaned_df.loc[idx, "patient_id"] if "patient_id" in cleaned_df.columns else str(idx)
                if pd.notna(val):
                    if val < 0 or val > 5:
                        issues.append({
                            "row_id": row_id,
                            "field": "previous_toxicity_grade",
                            "issue_type": "invalid_ctcae_grade",
                            "original_value": val,
                            "action_taken": "clamped_to_grade_range"
                        })
                        cleaned_df.loc[idx, "previous_toxicity_grade"] = min(5.0, max(0.0, float(val)))

        if "toxicity_risk" in cleaned_df.columns:
            valid_risks = {"Low", "Moderate", "High", "Critical"}
            for idx, val in cleaned_df["toxicity_risk"].items():
                row_id = cleaned_df.loc[idx, "patient_id"] if "patient_id" in cleaned_df.columns else str(idx)
                if pd.isna(val):
                    issues.append({
                        "row_id": row_id,
                        "field": "toxicity_risk",
                        "issue_type": "missing_toxicity_risk",
                        "original_value": None,
                        "action_taken": "imputed_unknown"
                    })
                    cleaned_df.loc[idx, "toxicity_risk"] = "Moderate"
                elif str(val).capitalize() not in valid_risks:
                    issues.append({
                        "row_id": row_id,
                        "field": "toxicity_risk",
                        "issue_type": "unrecognized_risk_tier",
                        "original_value": val,
                        "action_taken": "normalized"
                    })
                    cleaned_df.loc[idx, "toxicity_risk"] = "Moderate"
                else:
                    cleaned_df.loc[idx, "toxicity_risk"] = str(val).capitalize()

        return cleaned_df, issues
