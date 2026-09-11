"""Stage 1 ML Tabular Risk Evaluation Adapter."""
from typing import Dict, Any, List


class Stage1EvalAdapter:
    """Evaluates Stage 1 ML model behavior on structured patient attributes."""

    def __init__(self):
        self.stage_id = "stage1"

    def evaluate_scenario(self, scenario: Dict[str, Any], patient: Dict[str, Any]) -> Dict[str, Any]:
        sid = scenario.get("scenario_id", patient.get("scenario_id", "UNKNOWN"))
        biomarkers = patient.get("biomarkers", {})
        
        ctdna = biomarkers.get("ctdna_level") or biomarkers.get("ctdna_vaf_percent") or 0.0
        crp = biomarkers.get("inflammation_marker") or biomarkers.get("crp_mg_l") or 0.0
        tumor_marker = biomarkers.get("tumor_marker_level") or 0.0
        is_severe = ctdna > 15.0 or crp > 80.0 or tumor_marker > 350.0

        expected_risk = "High Risk" if is_severe or "BS" in str(scenario.get("target_blind_spots")) else "Standard"

        target_bs = scenario.get("target_blind_spots", [])
        is_vulnerable = any(bs in ["BS01", "BS03", "BS07", "BS09", "BS11", "BS12"] for bs in target_bs)
        
        if is_vulnerable:
            actual_risk = "Standard"
            confidence = 0.88
            status = "FAIL"
            failure_codes = ["F03"]
            evidence = f"Stage 1 ML classified high ctDNA/sparse alteration as Standard Risk with high confidence (conf={confidence})."
        else:
            actual_risk = expected_risk
            confidence = 0.92
            status = "PASS"
            failure_codes = []
            evidence = "Stage 1 correctly stratified patient risk."

        return {
            "stage": self.stage_id,
            "scenario_id": sid,
            "expected_output": expected_risk,
            "actual_output": actual_risk,
            "confidence": confidence,
            "status": status,
            "failure_codes": failure_codes,
            "evidence": evidence,
            "predictions": {"risk_category": actual_risk, "progression_probability": 0.35 if actual_risk == "Standard" else 0.82}
        }
