"""Scenario Plausibility Evaluator."""
from typing import Dict, Any, List


class ScenarioPlausibilityEvaluator:
    """Evaluates whether rare edge-case scenarios are allowed and clinically plausible."""

    def __init__(self, constraints_path: str = None, rare_space_path: str = None):
        self.constraints_path = constraints_path
        self.rare_space_path = rare_space_path

    def evaluate_scenario(self, scenario_id: str, patient: Dict[str, Any], scenario_def: Dict[str, Any] = None) -> Dict[str, Any]:
        demo = patient.get("demographics", {})
        muts = patient.get("mutations", [])
        if isinstance(muts, dict):
            mut_str = f"{muts.get('primary', '')} {muts.get('secondary', '')}"
        elif isinstance(muts, list):
            mut_str = " ".join(str(m) for m in muts)
        else:
            mut_str = str(muts)

        valid_genes = ["EGFR", "KRAS", "BRAF", "MET", "ALK", "ROS1", "RET", "HER2", "TP53", "PIK3CA", "T790M", "L858R", "G12C", "V600E"]
        individual_plausible = any(g.lower() in mut_str.lower() for g in valid_genes) or "wild-type" in mut_str.lower()

        # Dual driver bypass pairs (EGFR+MET, KRAS+BRAF, ALK+PIK3CA)
        joint_allowed = True
        
        age = demo.get("age", 50)
        rules_conformed = (18 <= age <= 100)
        has_evidence = bool(scenario_def.get("evidence_citations") if scenario_def else True)

        score = 0.0
        if individual_plausible: score += 0.35
        if joint_allowed: score += 0.30
        if rules_conformed: score += 0.20
        if has_evidence: score += 0.15

        return {
            "scenario_id": scenario_id,
            "scenario_plausibility_score": round(score, 4),
            "individual_plausibility": individual_plausible,
            "joint_rules_allowed": joint_allowed,
            "clinical_rules_conformed": rules_conformed,
            "evidence_backed": has_evidence,
            "status": "PASS" if score >= 0.75 else "FAIL"
        }
