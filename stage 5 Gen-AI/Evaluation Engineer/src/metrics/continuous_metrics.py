"""Continuous variable distribution metrics (mean, median, quantiles, Wasserstein, KS)."""
import numpy as np
from typing import Dict, List, Any
from scipy.stats import wasserstein_distance, ks_2samp


def compute_mean_difference(gen_vals: List[float], ref_vals: List[float]) -> float:
    if not gen_vals or not ref_vals:
        return 0.0
    return float(abs(np.mean(gen_vals) - np.mean(ref_vals)))


def compute_median_difference(gen_vals: List[float], ref_vals: List[float]) -> float:
    if not gen_vals or not ref_vals:
        return 0.0
    return float(abs(np.median(gen_vals) - np.median(ref_vals)))


def compute_std_difference(gen_vals: List[float], ref_vals: List[float]) -> float:
    if len(gen_vals) < 2 or len(ref_vals) < 2:
        return 0.0
    return float(abs(np.std(gen_vals, ddof=1) - np.std(ref_vals, ddof=1)))


def compute_quantile_differences(
    gen_vals: List[float], ref_vals: List[float], quantiles: List[float] = [0.05, 0.25, 0.50, 0.75, 0.95]
) -> Dict[str, float]:
    if not gen_vals or not ref_vals:
        return {f"p{int(q*100):02d}": 0.0 for q in quantiles}
    res = {}
    for q in quantiles:
        g_q = float(np.quantile(gen_vals, q))
        r_q = float(np.quantile(ref_vals, q))
        res[f"p{int(q*100):02d}"] = round(abs(g_q - r_q), 4)
    return res


def compute_wasserstein_distance(gen_vals: List[float], ref_vals: List[float]) -> float:
    if not gen_vals or not ref_vals:
        return 0.0
    return float(round(wasserstein_distance(gen_vals, ref_vals), 4))


def compute_ks_statistic(gen_vals: List[float], ref_vals: List[float]) -> Dict[str, float]:
    if not gen_vals or not ref_vals:
        return {"statistic": 0.0, "p_value": 1.0}
    ks_res = ks_2samp(gen_vals, ref_vals)
    return {
        "statistic": float(round(ks_res.statistic, 4)),
        "p_value": float(round(ks_res.pvalue, 4))
    }
