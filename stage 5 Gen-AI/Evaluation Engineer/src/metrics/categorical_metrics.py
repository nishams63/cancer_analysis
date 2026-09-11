"""Categorical variable distribution metrics (TVD, Jensen-Shannon, frequency diffs)."""
import numpy as np
from typing import Dict, List, Any
from scipy.spatial.distance import jensenshannon


def compute_frequency_difference(gen_counts: Dict[str, int], ref_counts: Dict[str, int]) -> Dict[str, float]:
    total_gen = max(1, sum(gen_counts.values()))
    total_ref = max(1, sum(ref_counts.values()))
    all_keys = set(gen_counts.keys()).union(ref_counts.keys())
    diffs = {}
    for k in sorted(all_keys):
        p_gen = gen_counts.get(k, 0) / total_gen
        p_ref = ref_counts.get(k, 0) / total_ref
        diffs[k] = round(abs(p_gen - p_ref), 4)
    return diffs


def compute_total_variation_distance(gen_probs: Dict[str, float], ref_probs: Dict[str, float]) -> float:
    all_keys = set(gen_probs.keys()).union(ref_probs.keys())
    tvd = 0.5 * sum(abs(gen_probs.get(k, 0.0) - ref_probs.get(k, 0.0)) for k in all_keys)
    return float(round(tvd, 4))


def compute_jensen_shannon_divergence(gen_probs: Dict[str, float], ref_probs: Dict[str, float]) -> float:
    all_keys = sorted(set(gen_probs.keys()).union(ref_probs.keys()))
    if not all_keys:
        return 0.0
    p = np.array([gen_probs.get(k, 0.0) for k in all_keys], dtype=float)
    q = np.array([ref_probs.get(k, 0.0) for k in all_keys], dtype=float)
    # Normalize
    if p.sum() > 0:
        p = p / p.sum()
    else:
        p = np.ones_like(p) / len(p)
    if q.sum() > 0:
        q = q / q.sum()
    else:
        q = np.ones_like(q) / len(q)
    return float(round(jensenshannon(p, q, base=2), 4))


def compute_category_coverage(gen_cats: List[str], ref_cats: List[str]) -> float:
    if not ref_cats:
        return 1.0
    covered = len(set(gen_cats).intersection(set(ref_cats)))
    return float(round(covered / len(set(ref_cats)), 4))
