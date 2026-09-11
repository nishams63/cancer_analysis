"""Scenario Fidelity Evaluator."""
from typing import Dict, Any, List


class ScenarioFidelityEvaluator:
    """Evaluates Level 3: Did Stage 5 create the exact scenario requested by EDA / Prompt Engineer?"""

    def __init__(self):
        pass

    def evaluate_fidelity(self, scenario_def: Dict[str, Any], patient: Dict[str, Any]) -> Dict[str, Any]:
        sid = scenario_def.get("scenario_id", patient.get("scenario_id", "UNKNOWN"))
        demo = patient.get("demographics", {})
        muts = patient.get("mutations", [])
        bios = patient.get("biomarkers", {})
        skel = scenario_def.get("patient_skeleton", {})

        satisfied_conditions = 0
        total_conditions = 0

        if "age" in skel:
            total_conditions += 1
            if abs(demo.get("age", 0) - skel["age"]) <= 2:
                satisfied_conditions += 1

        if "sex" in skel:
            total_conditions += 1
            if demo.get("sex") == skel["sex"]:
                satisfied_conditions += 1

        if "cancer_type" in skel:
            total_conditions += 1
            if demo.get("cancer_type") == skel["cancer_type"]:
                satisfied_conditions += 1

        mut_str = " ".join(muts).lower() if isinstance(muts, list) else str(muts).lower()

        req_entities = scenario_def.get("required_entities", [])
        for ent in req_entities:
            total_conditions += 1
            cat = ent.get("category", "")
            vals = [v.lower() for v in ent.get("values", [])]
            
            matched = False
            if cat == "mutation":
                if any(v in mut_str or any(word in mut_str for word in v.split()) for v in vals):
                    matched = True
            elif cat == "biomarker":
                pt_bios = str(bios).lower()
                if any(v in pt_bios or any(word in pt_bios for word in v.split()) for v in vals):
                    matched = True
            elif cat == "treatment":
                pt_tx = str(patient.get("treatments", [])).lower()
                if any(v in pt_tx or any(word in pt_tx for word in v.split()) for v in vals):
                    matched = True
            else:
                matched = True

            if matched:
                satisfied_conditions += 1

        forbidden_violations = []
        forbidden_mods = scenario_def.get("forbidden_modifications", [])
        pt_full_str = str(patient).lower()
        for fmod in forbidden_mods:
            fpattern = fmod.get("forbidden_pattern", "").lower()
            if fpattern and fpattern in pt_full_str:
                forbidden_violations.append(fmod.get("rule_id", "FORBID_VIOLATION"))

        base_compliance = satisfied_conditions / max(1, total_conditions)
        has_forbidden = len(forbidden_violations) > 0
        status = "FAIL" if (has_forbidden or base_compliance < 0.80) else "PASS"

        return {
            "scenario_id": sid,
            "fidelity_score": round(base_compliance, 4),
            "satisfied_conditions": satisfied_conditions,
            "total_conditions": total_conditions,
            "forbidden_violations": forbidden_violations,
            "status": status
        }
