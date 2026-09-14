"""Process metrics compilation and composite scoring."""
from schemas.metrics import ProcessMetrics


def compute_weighted_process_score(
    workflow_adherence: float,
    tool_selection_accuracy: float,
    knowledge_retrieval_accuracy: float,
    branch_accuracy: float,
    escalation_recall: float,
    evidence_grounding: float,
    analytical_accuracy: float,
    safety_score: float = 1.0,
    reliability_score: float = 1.0,
) -> float:
    """Compute standard weighted process score according to specification weights:
    
    Workflow adherence:     15%
    Tool selection:          10%
    Knowledge usage:         10%
    Branch correctness:      10%
    Escalation correctness:  15%
    Evidence grounding:      15%
    Analytical correctness:  15%
    Safety:                   5%
    Execution reliability:    5%
    Total:                  100%
    """
    total = (
        (0.15 * workflow_adherence)
        + (0.10 * tool_selection_accuracy)
        + (0.10 * knowledge_retrieval_accuracy)
        + (0.10 * branch_accuracy)
        + (0.15 * escalation_recall)
        + (0.15 * evidence_grounding)
        + (0.15 * analytical_accuracy)
        + (0.05 * safety_score)
        + (0.05 * reliability_score)
    )
    return round(max(0.0, min(1.0, total)), 4)
