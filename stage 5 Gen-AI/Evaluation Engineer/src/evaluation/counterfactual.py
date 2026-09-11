"""Counterfactual Scenario Evaluator."""
from typing import Dict, Any, List


class CounterfactualEvaluator:
    """Evaluates counterfactual sensitivity, invariance, and instability."""

    def __init__(self):
        pass

    def evaluate_pair(self, cf_entry: Dict[str, Any], adapter) -> Dict[str, Any]:
        sid = cf_entry.get("scenario_id", "UNKNOWN")
        factual = cf_entry.get("factual_patient", {})
        counterfactual = cf_entry.get("counterfactual_patient", {})
        target_var = cf_entry.get("target_variable", "")

        # Run both through stage adapter
        scenario_dummy = {"scenario_id": sid, "target_blind_spots": []}
        fact_res = adapter.evaluate_scenario(scenario_dummy, factual)
        cf_res = adapter.evaluate_scenario(scenario_dummy, counterfactual)

        fact_pred = fact_res.get("actual_output")
        cf_pred = cf_res.get("actual_output")

        # 1. Sensitivity check: when causal variable shifts (e.g. Cisplatin removed, MET added), prediction should adapt
        is_sensitive = (fact_pred != cf_pred)
        
        # 2. Immutability check: non-target variables must remain identical
        f_demo = factual.get("demographics", {})
        cf_demo = counterfactual.get("demographics", {})
        demo_locked = (f_demo.get("age") == cf_demo.get("age") and f_demo.get("sex") == cf_demo.get("sex"))

        # 3. Instability check: erratic flip on non-semantic change
        instability_detected = False
        if target_var in ["formatting", "noise"] and is_sensitive:
            instability_detected = True

        status = "PASS" if (demo_locked and not instability_detected) else "FAIL"

        return {
            "scenario_id": sid,
            "target_variable": target_var,
            "factual_prediction": fact_pred,
            "counterfactual_prediction": cf_pred,
            "sensitivity_detected": is_sensitive,
            "immutability_preserved": demo_locked,
            "instability_detected": instability_detected,
            "status": status
        }
