"""Outcome metrics compilation."""
from schemas.metrics import OutcomeMetrics


def compute_composite_outcome_score(
    analytical_accuracy: float,
    evidence_grounding: float,
    recommendation_relevance: float,
) -> float:
    """Compute overall outcome score:
    
    Analytical accuracy:      40%
    Evidence grounding:       30%
    Recommendation relevance: 30%
    """
    total = (0.40 * analytical_accuracy) + (0.30 * evidence_grounding) + (0.30 * recommendation_relevance)
    return round(max(0.0, min(1.0, total)), 4)
