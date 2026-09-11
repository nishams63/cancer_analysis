"""Treatment and regimen normalization."""
import pandas as pd
from typing import Any

class TreatmentNormalizer:
    DRUG_SYNONYMS = {
        "docetaxel": "Docetaxel",
        "taxotere": "Docetaxel",
        "osimertinib": "Osimertinib",
        "tagrisso": "Osimertinib",
        "cisplatin": "Cisplatin",
        "pembrolizumab": "Pembrolizumab",
        "keytruda": "Pembrolizumab",
        "trastuzumab": "Trastuzumab",
        "herceptin": "Trastuzumab",
        "radiotherapy-standard": "radiotherapy-standard",
        "radiation": "radiotherapy-standard"
    }

    def normalize_drug(self, val: Any) -> str:
        if pd.isna(val):
            return "Unknown"
        s = str(val).strip().lower()
        return self.DRUG_SYNONYMS.get(s, str(val).strip())

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "drug_name" in out.columns:
            out["drug_name"] = out["drug_name"].apply(self.normalize_drug)
        return out
