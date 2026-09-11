"""I/O helper utilities for loading and saving data files."""
import os
import json
import yaml
import pandas as pd
from pathlib import Path
from typing import Any, Dict

def ensure_parent_dir(filepath: str | Path) -> Path:
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def load_yaml(filepath: str | Path) -> Dict[str, Any]:
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"YAML file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def save_yaml(data: Dict[str, Any], filepath: str | Path) -> None:
    p = ensure_parent_dir(filepath)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, indent=2, allow_unicode=True)

def load_json(filepath: str | Path) -> Dict[str, Any]:
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: Any, filepath: str | Path, indent: int = 2) -> None:
    p = ensure_parent_dir(filepath)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)

def read_parquet(filepath: str | Path) -> pd.DataFrame:
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Parquet file not found: {p}")
    return pd.read_parquet(p)

def write_parquet(df: pd.DataFrame, filepath: str | Path, compression: str = "snappy") -> None:
    p = ensure_parent_dir(filepath)
    df.to_parquet(p, compression=compression, index=False)
