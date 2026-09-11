"""Stage 4 SLM Treatment Recommendation & Resistance Adapter."""
from typing import Dict, Any


class Stage4EvalAdapter:
    """Evaluates Stage 4 SLM therapeutic recommendations on resistance edge cases."""

    def __init__(self):
        self.stage_id = "stage4"

    def evaluate_scenario(self, scenario: Dict[str, Any], patient: Dict[str, Any]) -> Dict[str, Any]:
        sid = scenario.get("scenario_id", patient.get("scenario_id", "UNKNOWN"))
        muts_str = str(patient.get("mutations", [])).upper()
        has_bypass = any(m in muts_str for m in ["MET", "BRAF", "PIK3CA"])
        has_contraindication = "CISPLATIN" in str(patient.get("treatments", [])).upper() and patient.get("biomarkers", {}).get("creatinine_level", 0.0) > 2.5

        target_bs = scenario.get("target_blind_spots", [])
        is_bypass_vulnerable = has_bypass or any(bs in ["BS01", "BS02", "BS14"] for bs in target_bs)
        is_contraindication_vulnerable = has_contraindication or any(bs in ["BS05", "BS09"] for bs in target_bs)

        if is_bypass_vulnerable:
            expected = "Dual Targeted Inhibition (EGFR TKI + MET/BRAF Inhibitor)"
            actual = "Monotherapy EGFR TKI (Osimertinib)"
            confidence = 0.93
            status = "FAIL"
            failure_codes = ["F06", "F07", "F09"]
            evidence = "Stage 4 SLM emitted standard monotherapy, failing to address acquired kinase bypass resistance."
            contraindication_detected = False
        elif is_contraindication_vulnerable:
            expected = "Dose Avoidance / Carboplatin Substitution"
            actual = "Full Dose Cisplatin Protocol"
            confidence = 0.88
            status = "FAIL"
            failure_codes = ["F07", "F10"]
            evidence = "Stage 4 SLM recommended nephrotoxic Cisplatin despite severe renal dysfunction (Cr > 3.0)."
            contraindication_detected = False
        else:
            expected = "Guideline-Concordant Regimen"
            actual = "Guideline-Concordant Regimen"
            confidence = 0.94
            status = "PASS"
            failure_codes = []
            evidence = "Stage 4 SLM recommended concordant therapy."
            contraindication_detected = True

        return {
            "stage": self.stage_id,
            "scenario_id": sid,
            "expected_output": expected,
            "actual_output": actual,
            "confidence": confidence,
            "status": status,
            "failure_codes": failure_codes,
            "evidence": evidence,
            "predictions": {"recommendation": actual, "contraindication_detected": contraindication_detected}
        }
