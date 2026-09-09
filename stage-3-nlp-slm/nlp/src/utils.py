"""
Utility functions for Stage 3 Clinical NLP.
Provides cryptographic checksum computation, random seed management, and file serialization.
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, Any
import numpy as np


def compute_sha256(filepath: Path) -> str:
    """Calculate SHA256 cryptographic hash of a file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Cannot compute hash for missing file: {filepath}")
    return hashlib.sha256(filepath.read_bytes()).hexdigest()


def set_global_seed(seed: int = 42) -> None:
    """Enforce deterministic random seeds across standard library and numpy."""
    random.seed(seed)
    np.random.seed(seed)


def save_json(data: Any, filepath: Path, indent: int = 2) -> None:
    """Save serializable data to JSON file with UTF-8 encoding."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def load_json(filepath: Path) -> Any:
    """Load JSON file with UTF-8 encoding."""
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file missing at: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def capture_directory_hashes(directory: Path, pattern: str = "*.*") -> Dict[str, str]:
    """Capture SHA256 hashes of all files in a directory matching pattern."""
    hashes = {}
    for f in sorted(directory.glob(pattern)):
        if f.is_file():
            hashes[f.name] = compute_sha256(f)
    return hashes
