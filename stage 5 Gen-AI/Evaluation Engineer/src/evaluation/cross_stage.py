"""Cross-Stage End-to-End Stress Test Evaluator."""
from typing import Dict, Any, List
from ..adapters import Stage1EvalAdapter, Stage2EvalAdapter, Stage3EvalAdapter, Stage4EvalAdapter
from ..metrics.confidence_metrics import compute_confidence_spread
from ..metrics.disagreement_metrics import calculate_discordance_rate, build_pairwise_agreement_matrix


class CrossStageEvaluator:
    """Runs scenarios through Stages 1 to 4 and constructs cross-stage disagreement matrix."""

    def __init__(self):
        self.stage1 = Stage1EvalAdapter()
        self.stage2 = Stage2EvalAdapter()
        self.stage3 = Stage3EvalAdapter()
        self.stage4 = Stage4EvalAdapter()

    def evaluate_scenario(
        self,
        scenario: Dict[str, Any],
        patient: Dict[str, Any],
        narrative: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        sid = scenario.get("scenario_id", "UNKNOWN")
        
        r1 = self.stage1.evaluate_scenario(scenario, patient)
        r2 = self.stage2.evaluate_scenario(scenario, patient)
        r3 = self.stage3.evaluate_scenario(scenario, patient, narrative)
        r4 = self.stage4.evaluate_scenario(scenario, patient)

        stage_results = {"stage1": r1, "stage2": r2, "stage3": r3, "stage4": r4}

        # Measure confidence spread
        confs = [r["confidence"] for r in stage_results.values()]
        conf_spread = compute_confidence_spread(confs)

        # Measure failures
        failed_stages = [st for st, r in stage_results.items() if r["status"] == "FAIL"]
        all_failure_codes = []
        for r in stage_results.values():
            all_failure_codes.extend(r.get("failure_codes", []))
        all_failure_codes = sorted(list(set(all_failure_codes)))

        # Disagreement flag (e.g. Stage 1 says Standard, but Stage 3 says High Urgency or Stage 4 fails)
        has_disagreement = len(failed_stages) > 0 and len(failed_stages) < 4

        return {
            "scenario_id": sid,
            "stage_results": stage_results,
            "failed_stages": failed_stages,
            "failure_count": len(failed_stages),
            "all_failure_codes": all_failure_codes,
            "confidence_spread": round(conf_spread, 4),
            "has_cross_stage_disagreement": has_disagreement
        }

    def summarize_batch(self, cross_stage_evals: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(cross_stage_evals)
        if total == 0:
            return {}

        predictions_list = []
        for cs in cross_stage_evals:
            s_dict = {}
            for s_name, res in cs["stage_results"].items():
                s_dict[s_name] = res.get("predictions", {})
            predictions_list.append({"predictions": s_dict})

        discordance = calculate_discordance_rate(predictions_list)
        pairwise = build_pairwise_agreement_matrix([p["predictions"] for p in predictions_list])

        stage_fail_rates = {
            "stage1": sum(1 for cs in cross_stage_evals if "stage1" in cs["failed_stages"]) / total,
            "stage2": sum(1 for cs in cross_stage_evals if "stage2" in cs["failed_stages"]) / total,
            "stage3": sum(1 for cs in cross_stage_evals if "stage3" in cs["failed_stages"]) / total,
            "stage4": sum(1 for cs in cross_stage_evals if "stage4" in cs["failed_stages"]) / total,
        }

        return {
            "total_scenarios": total,
            "discordance_rate": discordance,
            "pairwise_agreement": pairwise,
            "stage_failure_rates": stage_fail_rates
        }
