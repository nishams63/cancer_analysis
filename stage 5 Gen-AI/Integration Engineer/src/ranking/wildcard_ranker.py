from typing import Dict, Any, List, Optional
from .candidate_filter import CandidateFilter
from .ranking_explainer import RankingExplainer
from ..utils.logging import get_logger

logger = get_logger("WildcardRanker")

DEFAULT_WEIGHTS = {
    "plausibility": 0.10,
    "fidelity": 0.10,
    "difficulty": 0.20,
    "impact": 0.25,
    "multi_stage_failure": 0.15,
    "reproducibility": 0.10,
    "novelty": 0.10
}

class WildcardRanker:
    def __init__(self, weights: Optional[Dict[str, float]] = None, filter_rules: Optional[CandidateFilter] = None):
        self.weights = weights or DEFAULT_WEIGHTS
        self.filter = filter_rules or CandidateFilter()
        self.explainer = RankingExplainer()

    def score_candidate(self, record: Dict[str, Any]) -> float:
        eval_res = record.get("evaluation", {})
        stages = record.get("stages", {})
        failures = record.get("failures", [])
        
        plaus = float(eval_res.get("scenario_plausibility", {}).get("score", 0.85)) * 100.0
        fid = float(eval_res.get("fidelity", {}).get("fidelity_score", 1.0)) * 100.0
        diff = float(eval_res.get("difficulty", {}).get("difficulty_score", 50.0))
        imp = float(eval_res.get("impact", {}).get("impact_score", 50.0))
        
        # Multi-stage failure factor
        failed_stages = set()
        for f in failures:
            failed_stages.add(f.get("stage"))
        multi_stage_score = min(100.0, len(failed_stages) * 33.33)

        # Reproducibility factor
        repro_score = 100.0 if record.get("batch_id") else 50.0

        # Novelty factor (e.g. rare archetype or counterfactual instability)
        cf = record.get("counterfactuals", [])
        has_instability = any(c.get("instability_detected", False) for c in cf)
        novelty_score = 90.0 if has_instability else 70.0

        score = (
            self.weights.get("plausibility", 0.10) * plaus +
            self.weights.get("fidelity", 0.10) * fid +
            self.weights.get("difficulty", 0.20) * diff +
            self.weights.get("impact", 0.25) * imp +
            self.weights.get("multi_stage_failure", 0.15) * multi_stage_score +
            self.weights.get("reproducibility", 0.10) * repro_score +
            self.weights.get("novelty", 0.10) * novelty_score
        )
        return round(score, 2)

    def rank_candidates(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        eligible = self.filter.filter_candidates(records)
        scored = []
        for r in eligible:
            sc = self.score_candidate(r)
            scored.append((sc, r))

        # Sort descending by composite score
        scored.sort(key=lambda x: x[0], reverse=True)

        ranked = []
        for i, (score, r) in enumerate(scored, 1):
            expl = self.explainer.explain(r, score, i)
            ranked.append({
                "rank": i,
                "scenario_id": r.get("scenario_id"),
                "batch_id": r.get("batch_id"),
                "wildcard_score": score,
                "is_candidate": True,
                "status_label": "CANDIDATE ONLY",
                "reason_explanation": expl,
                "record": r
            })
        return ranked
