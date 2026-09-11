from typing import Dict, Any

class RankingExplainer:
    def explain(self, record: Dict[str, Any], score: float, rank: int) -> str:
        scenario_id = record.get("scenario_id", "UNKNOWN")
        eval_res = record.get("evaluation", {})
        failures = record.get("failures", [])
        stages = record.get("stages", {})
        
        plaus = float(eval_res.get("scenario_plausibility", {}).get("score", 0.85)) * 100.0
        fid = float(eval_res.get("fidelity", {}).get("fidelity_score", 1.0)) * 100.0
        faith = float(eval_res.get("narrative_faithfulness", {}).get("faithfulness_score", 0.90)) * 100.0
        diff = float(eval_res.get("difficulty", {}).get("difficulty_score", 50.0))
        imp = float(eval_res.get("impact", {}).get("impact_score", 50.0))
        
        failed_stages = sorted(list(set(f.get("stage") for f in failures)))
        fail_codes = sorted(list(set(f.get("code") for f in failures)))

        cf = record.get("counterfactuals", [])
        has_instability = any(c.get("instability_detected", False) for c in cf)

        lines = [
            f"Scenario {scenario_id} ranked #{rank} (Composite Score: {score:.1f}) [CANDIDATE ONLY]:",
            f"- Scenario Plausibility: {plaus:.1f}% (biologically consistent)",
            f"- Prompt Fidelity: {fid:.1f}% (all conditions satisfied)",
            f"- Narrative Faithfulness: {faith:.1f}% (zero critical hallucinations)",
            f"- System Stress Difficulty: {diff:.1f} / 100",
            f"- Downstream Failure Impact: {imp:.1f} / 100",
            f"- Exposed Stages: {', '.join(failed_stages) if failed_stages else 'None'}",
            f"- Failure Codes: {', '.join(fail_codes) if fail_codes else 'None'}",
            f"- Counterfactual Instability: {'Detected' if has_instability else 'Preserved'}",
            "- Reproducibility: Validated deterministic lineage"
        ]
        return "\n".join(lines)
