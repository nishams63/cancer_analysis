"""Categorical normalization for cancer stages, risk tiers, and organ hazards."""
import pandas as pd
from typing import Any

class CategoryNormalizer:
    STAGE_MAP = {
        "stage i": "Stage I", "stage 1": "Stage I", "i": "Stage I",
        "stage ii": "Stage II", "stage 2": "Stage II", "ii": "Stage II",
        "stage iii": "Stage III", "stage 3": "Stage III", "iii": "Stage III",
        "stage iv": "Stage IV", "stage 4": "Stage IV", "iv": "Stage IV"
    }

    HAZARD_MAP = {
        "none": "NONE",
        "hepatic": "HEPATIC",
        "pulmonary": "PULMONARY",
        "hematologic": "HEMATOLOGIC",
        "renal": "RENAL",
        "neuropathic": "NEUROPATHIC",
        "dermatologic": "DERMATOLOGIC",
        "cardiac": "CARDIAC"
    }

    def normalize_stage(self, val: Any) -> str:
        if pd.isna(val):
            return "Unknown"
        s = str(val).strip().lower()
        return self.STAGE_MAP.get(s, str(val).strip())

    def normalize_hazard(self, val: Any) -> str:
        if pd.isna(val):
            return "NONE"
        s = str(val).strip().lower()
        return self.HAZARD_MAP.get(s, "NONE")

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "cancer_stage" in out.columns:
            out["cancer_stage"] = out["cancer_stage"].apply(self.normalize_stage)
        if "hazard_type" in out.columns:
            out["hazard_type"] = out["hazard_type"].apply(self.normalize_hazard)
        return out
