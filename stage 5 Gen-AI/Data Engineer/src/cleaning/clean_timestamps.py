"""Timestamp and chronological interval cleaning."""
from typing import Dict, Any, List, Tuple
import pandas as pd

class TimestampCleaner:
    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        cleaned_df = df.copy()
        issues = []

        date_cols = [c for c in ["observation_date", "document_date", "timestamp"] if c in cleaned_df.columns]
        for col in date_cols:
            for idx, val in cleaned_df[col].items():
                row_id = cleaned_df.loc[idx, "patient_id"] if "patient_id" in cleaned_df.columns else str(idx)
                if pd.notna(val):
                    try:
                        parsed = pd.to_datetime(val)
                        cleaned_df.loc[idx, col] = parsed.strftime("%Y-%m-%d")
                    except Exception as e:
                        issues.append({
                            "row_id": row_id,
                            "field": col,
                            "issue_type": "unparseable_timestamp",
                            "original_value": val,
                            "action_taken": "set_to_none"
                        })
                        cleaned_df.loc[idx, col] = None

        if "delta_days" in cleaned_df.columns:
            for idx, val in cleaned_df["delta_days"].items():
                row_id = cleaned_df.loc[idx, "patient_id"] if "patient_id" in cleaned_df.columns else str(idx)
                if pd.notna(val) and val < 0:
                    issues.append({
                        "row_id": row_id,
                        "field": "delta_days",
                        "issue_type": "negative_delta_days",
                        "original_value": val,
                        "action_taken": "reset_to_zero"
                    })
                    cleaned_df.loc[idx, "delta_days"] = 0

        return cleaned_df, issues
