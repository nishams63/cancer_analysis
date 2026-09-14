"""Multi-Factor Analytical Conflict Resolution Engine."""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ConflictResolutionResult(BaseModel):
    """Outcome of multi-factor competing candidate cause evaluation."""
    ranked_causes: List[Dict[str, Any]]
    top_cause: Dict[str, Any]
    second_cause: Optional[Dict[str, Any]] = None
    cause_score_margin: float
    is_ambiguous: bool
    requires_human_escalation: bool
    recommendation: str

    def __iter__(self):
        return iter((self.top_cause, self.cause_score_margin, self.requires_human_escalation))


def resolve_competing_causes(
    candidates: List[Dict[str, Any]],
    ambiguity_margin: float = 0.10,
) -> ConflictResolutionResult:
    """Deterministically score and rank competing root cause candidates.
    
    Formula:
      Score = 0.30 * effect_size + 0.25 * sample_size_norm + 0.25 * evidence_strength + 0.20 * consistency
    """
    if not candidates:
        empty_cause = {"cause": "Unknown", "score": 0.0}
        return ConflictResolutionResult(
            ranked_causes=[],
            top_cause=empty_cause,
            cause_score_margin=0.0,
            is_ambiguous=True,
            requires_human_escalation=True,
            recommendation="No candidate causes provided. Escalate to human review.",
        )

    scored = []
    for cand in candidates:
        name = cand.get("cause") or cand.get("factor") or cand.get("name") or "Unknown Factor"
        effect = float(cand.get("effect_size", 0.70))
        
        # Sample size normalized (saturation at 10,000)
        raw_n = float(cand.get("sample_size", 10000))
        sample_norm = min(1.0, max(0.1, raw_n / 10000.0))
        
        evid = float(cand.get("evidence_strength", 0.75))
        consist = float(cand.get("consistency", 0.75))

        score = round((0.30 * effect) + (0.25 * sample_norm) + (0.25 * evid) + (0.20 * consist), 4)

        item = dict(cand)
        item["name"] = name
        item["overall_score"] = score
        scored.append(item)

    scored.sort(key=lambda x: x["overall_score"], reverse=True)
    top = scored[0]
    second = scored[1] if len(scored) > 1 else None

    if second:
        margin = round(top["overall_score"] - second["overall_score"], 4)
    else:
        margin = round(top["overall_score"], 4)

    is_ambiguous = (second is not None) and (margin < ambiguity_margin)

    if is_ambiguous:
        rec = (
            f"Ambiguous competing root causes: '{top['name']}' (score: {top['overall_score']}) and "
            f"'{second['name']}' (score: {second['overall_score']}) have narrow margin ({margin} < {ambiguity_margin}). "
            f"Escalation required."
        )
    else:
        rec = f"Primary empirical root cause established: '{top['name']}' (margin: {margin})."

    return ConflictResolutionResult(
        ranked_causes=scored,
        top_cause=top,
        second_cause=second,
        cause_score_margin=margin,
        is_ambiguous=is_ambiguous,
        requires_human_escalation=is_ambiguous,
        recommendation=rec,
    )
