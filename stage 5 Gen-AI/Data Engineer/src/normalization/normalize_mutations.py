"""Ontology-based mutation normalization."""
import pandas as pd
from typing import Dict, Any, List

class MutationNormalizer:
    # Explicit clinical oncology ontology dictionary
    SYNONYM_MAP = {
        "EGFR-MUT": "EGFR",
        "EGFR+": "EGFR",
        "EGFR MUTANT": "EGFR",
        "KRAS-MUT": "KRAS",
        "KRAS+": "KRAS",
        "TP53-MUT": "TP53",
        "TP53+": "TP53",
        "BRAF V600E": "BRAF",
        "BRAF-MUT": "BRAF",
        "ALK FUSION": "ALK",
        "ALK+": "ALK",
        "MET AMP": "MET",
        "MET AMPLIFICATION": "MET",
        "NONE": "None/Unknown",
        "UNKNOWN": "None/Unknown",
        "WILDTYPE": "None/Unknown"
    }

    def normalize_value(self, val: Any) -> str:
        if pd.isna(val):
            return "None/Unknown"
        s = str(val).strip().upper()
        return self.SYNONYM_MAP.get(s, s)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in ["mutation_primary", "mutation_secondary"]:
            if col in out.columns:
                out[col] = out[col].apply(self.normalize_value)
        return out
