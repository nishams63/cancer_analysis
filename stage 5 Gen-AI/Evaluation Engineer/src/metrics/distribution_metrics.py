"""Joint distribution & co-occurrence consistency metrics."""
from typing import Dict, List, Any, Set, Tuple
import pandas as pd


def check_cooccurrence_constraints(
    mutations: List[str],
    allowed_mutations: Set[str],
    forbidden_pairs: List[Tuple[str, str]]
) -> Dict[str, Any]:
    mut_set = set(mutations)
    # Check unsupported mutations
    unsupported = list(mut_set - allowed_mutations)
    
    # Check forbidden combinations
    violations = []
    for m1, m2 in forbidden_pairs:
        if m1 in mut_set and m2 in mut_set:
            violations.append(f"{m1} + {m2}")
            
    is_valid = len(unsupported) == 0 and len(violations) == 0
    return {
        "is_valid": is_valid,
        "unsupported_mutations": unsupported,
        "forbidden_pair_violations": violations
    }


def compute_joint_frequency_divergence(
    observed_joint_freq: float,
    reference_joint_freq: float
) -> float:
    return float(round(abs(observed_joint_freq - reference_joint_freq), 4))
