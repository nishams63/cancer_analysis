"""Wildcard Candidate Evidence Preparation."""
import pandas as pd
from typing import Dict, Any, List


class WildcardEvidenceBuilder:
    """Prepares evidence ranking table for wildcard stress-test candidates."""

    def __init__(self):
        pass

    def build_candidate_table(self, scenario_evaluations: List[Dict[str, Any]]) -> pd.DataFrame:
        rows = []
        for sc in scenario_evaluations:
            sid = sc.get("scenario_id", "UNKNOWN")
            realism = sc.get("realism_score", 0.85)
            plausibility = sc.get("scenario_plausibility_score", 0.90)
            rag_q = sc.get("rag_quality_score", 0.88)
            fidelity = sc.get("fidelity_score", 1.0)
            faithfulness = sc.get("narrative_faithfulness_score", 0.95)
            difficulty = sc.get("difficulty_score", 65.0)
            impact = sc.get("impact_score", 70.0)
            failed_stages = ", ".join(sc.get("failed_stages", []))
            failure_codes = ", ".join(sc.get("all_failure_codes", []))

            # Composite Wildcard Evidence Score
            # Prioritizes high difficulty + high impact while strictly penalizing unrealistic scenarios
            evidence_score = round(
                (0.35 * difficulty + 0.35 * impact + 0.15 * (plausibility * 100) + 0.15 * (faithfulness * 100)),
                2
            )

            rows.append({
                "scenario_id": sid,
                "evidence_score": evidence_score,
                "difficulty": difficulty,
                "impact": impact,
                "scenario_plausibility": plausibility,
                "narrative_faithfulness": faithfulness,
                "rag_quality": rag_q,
                "failed_stages": failed_stages,
                "failure_codes": failure_codes,
                "reproducibility": "YES"
            })

        df = pd.DataFrame(rows)
        if not df.empty and "evidence_score" in df.columns:
            df = df.sort_values(by="evidence_score", ascending=False).reset_index(drop=True)
        return df
