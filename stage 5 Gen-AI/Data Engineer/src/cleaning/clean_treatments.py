"""Treatment and dosage cleaning."""
from typing import Dict, Any, List, Tuple
import pandas as pd

class TreatmentCleaner:
    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        cleaned_df = df.copy()
        issues = []

        # Validate treatment cycles
        if "treatment_cycle" in cleaned_df.columns:
            for idx, val in cleaned_df["treatment_cycle"].items():
                row_id = cleaned_df.loc[idx, "patient_id"] if "patient_id" in cleaned_df.columns else str(idx)
                if pd.notna(val) and val < 0:
                    issues.append({
                        "row_id": row_id,
                        "field": "treatment_cycle",
                        "issue_type": "negative_cycle_count",
                        "original_value": val,
                        "action_taken": "reset_to_zero"
                    })
                    cleaned_df.loc[idx, "treatment_cycle"] = 0

        # Validate drug doses
        if "drug_dose" in cleaned_df.columns:
            for idx, val in cleaned_df["drug_dose"].items():
                row_id = cleaned_df.loc[idx, "patient_id"] if "patient_id" in cleaned_df.columns else str(idx)
                if pd.notna(val) and val < 0:
                    issues.append({
                        "row_id": row_id,
                        "field": "drug_dose",
                        "issue_type": "negative_dosage",
                        "original_value": val,
                        "action_taken": "flagged_and_zeroed"
                    })
                    cleaned_df.loc[idx, "drug_dose"] = 0.0

        # Treatment type standardization
        if "treatment_type" in cleaned_df.columns:
            cleaned_df["treatment_type"] = cleaned_df["treatment_type"].fillna("Unknown").astype(str).str.strip()

        return cleaned_df, issues
