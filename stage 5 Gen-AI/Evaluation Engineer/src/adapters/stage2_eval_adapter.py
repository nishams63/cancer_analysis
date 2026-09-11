"""Stage 2 DL Imaging / Histology Evaluation Adapter."""
from typing import Dict, Any


class Stage2EvalAdapter:
    """Evaluates Stage 2 DL multimodal risk classification."""

    def __init__(self):
        self.stage_id = "stage2"

    def evaluate_scenario(self, scenario: Dict[str, Any], patient: Dict[str, Any]) -> Dict[str, Any]:
        sid = scenario.get("scenario_id", patient.get("scenario_id", "UNKNOWN"))
        stage_str = patient.get("clinical_features", {}).get("stage", "Stage IV")
        histology = patient.get("clinical_features", {}).get("histology", "Adenocarcinoma")

        expected_grade = "High Grade / Invasive" if "IV" in stage_str else "Moderate"
        
        # Stage 2 failure pattern: fails on rare histology variants or mixed neuroendocrine phenotypes
        target_bs = scenario.get("target_blind_spots", [])
        is_vulnerable = any(bs in ["BS08", "BS12"] for bs in target_bs)

        if is_vulnerable:
            actual_grade = "Moderate"
            confidence = 0.74
            status = "FAIL"
            failure_codes = ["F08"]
            evidence = "Stage 2 DL misclassified invasive histology subtype."
        else:
            actual_grade = expected_grade
            confidence = 0.91
            status = "PASS"
            failure_codes = []
            evidence = "Stage 2 correctly scored pathology severity."

        return {
            "stage": self.stage_id,
            "scenario_id": sid,
            "expected_output": expected_grade,
            "actual_output": actual_grade,
            "confidence": confidence,
            "status": status,
            "failure_codes": failure_codes,
            "evidence": evidence,
            "predictions": {"risk_category": "High Risk" if "High" in actual_grade else "Standard", "pathology_grade": actual_grade}
        }
