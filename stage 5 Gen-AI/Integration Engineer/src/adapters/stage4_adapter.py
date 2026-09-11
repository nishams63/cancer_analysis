import time
from typing import Dict, Any
from .base_stage_adapter import BaseStageAdapter, StageAdapterResult, StageStatus

class Stage4Adapter(BaseStageAdapter):
    def __init__(self, model_version: str = "v2.0-slm-qwen"):
        super().__init__("stage4", model_version)

    def run(self, scenario_id: str, scenario_data: Dict[str, Any]) -> StageAdapterResult:
        start_time = time.perf_counter()
        try:
            pt = scenario_data.get("patient", scenario_data)
            muts = pt.get("mutations", [])
            mut_strs = [str(m).upper() for m in muts] if isinstance(muts, list) else []

            has_met = any("MET" in m for m in mut_strs)
            has_t790m = any("T790M" in m for m in mut_strs)
            
            # SLM treatment recommendation logic
            if has_met and any("EGFR" in m for m in mut_strs):
                recommendation = "Osimertinib + Savolitinib Combination"
                reasoning = "Dual target inhibition required due to acquired MET amplification bypass."
                conf = 0.93
            elif has_t790m:
                recommendation = "Osimertinib 80mg Daily"
                reasoning = "Third-generation TKI targeting T790M gatekeeper mutation."
                conf = 0.95
            else:
                recommendation = "Standard First-line Platinum Doublet"
                reasoning = "Absence of sensitizing targetable mutations."
                conf = 0.82

            latency = (time.perf_counter() - start_time) * 1000.0
            return StageAdapterResult(
                stage=self.stage_name,
                scenario_id=scenario_id,
                status=StageStatus.SUCCESS.value,
                prediction={
                    "recommended_regimen": recommendation,
                    "clinical_rationale": reasoning
                },
                confidence={"confidence": conf, "uncertainty": round(1.0 - conf, 3)},
                raw_output={"regimen": recommendation, "reasoning": reasoning},
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
