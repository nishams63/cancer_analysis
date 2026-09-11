"""Validation helpers for metrics and evaluation scores."""
from typing import Any


def validate_score_range(val: float, min_val: float = 0.0, max_val: float = 1.0, name: str = "Score") -> float:
    if not (min_val <= val <= max_val):
        raise ValueError(f"{name} must be in [{min_val}, {max_val}], got {val}")
    return val


def validate_non_empty(val: Any, name: str = "Object") -> Any:
    if not val:
        raise ValueError(f"{name} must not be empty or null")
    return val
