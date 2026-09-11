"""Cryptographic hashing utilities for files, strings, and DataFrames."""
import hashlib
from pathlib import Path
import pandas as pd

def compute_file_hash(filepath: str | Path, algorithm: str = "sha256") -> str:
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"File not found for hashing: {p}")
    h = hashlib.new(algorithm)
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def compute_string_hash(text: str, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    h.update(text.encode("utf-8"))
    return h.hexdigest()

def compute_dataframe_hash(df: pd.DataFrame, algorithm: str = "sha256") -> str:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    h = hashlib.new(algorithm)
    h.update(csv_bytes)
    return h.hexdigest()
