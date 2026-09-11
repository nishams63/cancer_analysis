"""I/O helpers for evaluation engine."""
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd


def load_yaml(filepath: str | Path) -> Dict[str, Any]:
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"YAML file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: Any, filepath: str | Path) -> None:
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_json(filepath: str | Path) -> Any:
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, filepath: str | Path, indent: int = 2) -> None:
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def load_jsonl(filepath: str | Path) -> List[Dict[str, Any]]:
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {p}")
    records = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line_s = line.strip()
            if line_s:
                records.append(json.loads(line_s))
    return records


def save_jsonl(records: List[Dict[str, Any]], filepath: str | Path) -> None:
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")


def read_parquet(filepath: str | Path) -> pd.DataFrame:
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Parquet file not found: {p}")
    return pd.read_parquet(p)


def write_parquet(df: pd.DataFrame, filepath: str | Path) -> None:
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
