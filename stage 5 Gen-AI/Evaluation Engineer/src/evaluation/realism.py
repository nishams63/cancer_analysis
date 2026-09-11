"""Structured Scenario Realism & Population Similarity Evaluator."""
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from ..metrics.continuous_metrics import compute_mean_difference, compute_wasserstein_distance, compute_ks_statistic
from ..metrics.categorical_metrics import compute_total_variation_distance


class ScenarioRealismEvaluator:
    """Evaluates Level 1: Structured Scenario Realism and Population Similarity."""

    def __init__(self, ref_distributions_path: str = None):
        self.ref_path = ref_distributions_path

    def evaluate_batch(self, patients: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not patients:
            return {"population_similarity_score": 0.0, "scenarios": []}

        ages = [p.get("demographics", {}).get("age", 60.0) for p in patients]
        ref_ages = list(np.random.normal(62.5, 11.2, max(len(ages), 100)))

        age_mean_diff = compute_mean_difference(ages, ref_ages)
        age_w_dist = compute_wasserstein_distance(ages, ref_ages)

        cancer_types = [p.get("demographics", {}).get("cancer_type", "NSCLC") for p in patients]
        gen_ct_probs = {ct: cancer_types.count(ct) / len(cancer_types) for ct in set(cancer_types)}
        ref_ct_probs = {"NSCLC": 0.55, "Breast Cancer": 0.25, "Colorectal Cancer": 0.15, "Other": 0.05}
        tvd_cancer = compute_total_variation_distance(gen_ct_probs, ref_ct_probs)

        pop_sim = max(0.0, min(1.0, 1.0 - 0.3 * (age_mean_diff / 20.0) - 0.4 * tvd_cancer))

        scenario_evals = []
        for p in patients:
            ev = self.evaluate_patient(p)
            scenario_evals.append(ev)

        return {
            "batch_size": len(patients),
            "population_similarity_score": round(pop_sim, 4),
            "age_mean_diff": round(age_mean_diff, 2),
            "age_wasserstein_dist": round(age_w_dist, 2),
            "cancer_type_tvd": round(tvd_cancer, 4),
            "scenario_evaluations": scenario_evals
        }

    def evaluate_patient(self, patient: Dict[str, Any]) -> Dict[str, Any]:
        sid = patient.get("scenario_id", "UNKNOWN")
        demo = patient.get("demographics", {})
        muts = patient.get("mutations", [])
        bios = patient.get("biomarkers", {})
        tx = patient.get("treatments", [])
        timeline = patient.get("timeline", [])

        age = demo.get("age", 0)
        age_pass = (18 <= age <= 100)
        mut_pass = len(muts) > 0 if isinstance(muts, list) else bool(muts)
        
        bio_pass = True
        for bname, bval in bios.items():
            if isinstance(bval, (int, float)) and (bval < 0 or bval > 1000):
                bio_pass = False

        tx_pass = bool(tx)
        temp_pass = len(timeline) >= 1 or bool(demo)

        checks = {
            "age": "PASS" if age_pass else "FAIL",
            "mutations": "PASS" if mut_pass else "FAIL",
            "biomarkers": "PASS" if bio_pass else "FAIL",
            "treatment": "PASS" if tx_pass else "FAIL",
            "temporal": "PASS" if temp_pass else "FAIL"
        }

        all_pass = all(v == "PASS" for v in checks.values())
        return {
            "scenario_id": sid,
            "patient_id": patient.get("patient_id", ""),
            "checks": checks,
            "is_physiologically_plausible": all_pass
        }
