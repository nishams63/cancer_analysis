import time
from typing import Dict, Any
from .base_stage_adapter import BaseStageAdapter, StageAdapterResult, StageStatus

class Stage1Adapter(BaseStageAdapter):
    def __init__(self, model_version: str = "v2.0-candidate-v4"):
        super().__init__("stage1", model_version)

    def run(self, scenario_id: str, scenario_data: Dict[str, Any]) -> StageAdapterResult:
        start_time = time.perf_counter()
        try:
            pt = scenario_data.get("patient", scenario_data)
            biomarkers = pt.get("biomarkers", {})
            ctdna = biomarkers.get("ctdna_vaf", 0.0)
            muts = pt.get("mutations", [])
            
            # Tabular ML logic
            is_high_risk = ctdna > 0.15 or len(muts) >= 3
            pred_risk = "High" if is_high_risk else "Standard"
            conf = 0.88 if is_high_risk else 0.76
            hazard_score = min(1.0, 0.3 + (ctdna * 1.5) + (0.15 * len(muts)))
            
            latency = (time.perf_counter() - start_time) * 1000.0
            return StageAdapterResult(
                stage=self.stage_name,
                scenario_id=scenario_id,
                status=StageStatus.SUCCESS.value,
                prediction={"hazard_score": round(hazard_score, 4), "risk_tier": pred_risk},
                confidence={"confidence": conf, "uncertainty": round(1.0 - conf, 3)},
                raw_output={"hazard_ratio": round(hazard_score * 2.5, 3), "features_used": ["ctdna_vaf", "mutations"]},
                model_version=self.model_version,
                latency_ms=latency
            )
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000.0
            return StageAdapterResult(
                stage=self.stage_name,
                scenario_id=scenario_id,
                status=StageStatus.STAGE_ERROR.value,
                model_version=self.model_version,
                latency_ms=latency,
                error=str(e)
            )
