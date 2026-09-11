"""Validation helpers for tabular and structural data."""
from typing import List, Dict, Any
import pandas as pd

def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    return True

def check_null_counts(df: pd.DataFrame) -> Dict[str, int]:
    return df.isnull().sum().to_dict()
