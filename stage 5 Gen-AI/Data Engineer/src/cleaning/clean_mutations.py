"""Mutation string cleaning and standardization."""
from typing import Dict, Any, List, Tuple
import pandas as pd

class MutationCleaner:
    ALLOWED_CANONICAL_MUTATIONS = {
        "KRAS", "EGFR", "TP53", "ALK", "MET", "BRAF", "PIK3CA", "HER2", "ROS1", "RET", "None/Unknown"
    }

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        cleaned_df = df.copy()
        issues = []

        for col in ["mutation_primary", "mutation_secondary"]:
            if col in cleaned_df.columns:
                for idx, val in cleaned_df[col].items():
                    row_id = cleaned_df.loc[idx, "patient_id"] if "patient_id" in cleaned_df.columns else str(idx)
                    if pd.isna(val) or str(val).strip().lower() in {"none", "unknown", "nan", ""}:
                        cleaned_df.loc[idx, col] = "None/Unknown"
                    else:
                        norm_val = str(val).strip().upper()
                        # Extract root gene symbol if prefixed/suffixed
                        matched = None
                        for gene in self.ALLOWED_CANONICAL_MUTATIONS:
                            if gene != "None/Unknown" and (norm_val == gene or norm_val.startswith(gene)):
                                matched = gene
                                break
                        if matched:
                            if matched != val:
                                issues.append({
                                    "row_id": row_id,
                                    "field": col,
                                    "issue_type": "inconsistent_mutation_syntax",
                                    "original_value": val,
                                    "action_taken": f"standardized_to_{matched}"
                                })
                            cleaned_df.loc[idx, col] = matched
                        else:
                            issues.append({
                                "row_id": row_id,
                                "field": col,
                                "issue_type": "unrecognized_mutation",
                                "original_value": val,
                                "action_taken": "preserved_with_flag"
                            })

        return cleaned_df, issues
