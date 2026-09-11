"""Cross-stage disagreement and discordance rate metrics."""
from typing import List, Dict, Any, Tuple


def calculate_sensitivity_drop(standard_sens: float, stress_sens: float) -> float:
    """Calculate absolute sensitivity degradation: Delta Sens = Sens_standard - Sens_stress."""
    return round(max(0.0, standard_sens - stress_sens), 4)


def calculate_discordance_rate(stage_predictions_list: List[Dict[str, Any]]) -> float:
    """Proportion of scenarios where stages disagree on risk classification."""
    if not stage_predictions_list:
        return 0.0
    disagree_count = 0
    for entry in stage_predictions_list:
        preds = entry.get("predictions", {})
        # Extract binary or categorical risk/urgency across stages
        risks = [p.get("risk_category") or p.get("urgency") for p in preds.values() if isinstance(p, dict)]
        risks = [r for r in risks if r is not None]
        if len(set(risks)) > 1:
            disagree_count += 1
    return round(disagree_count / len(stage_predictions_list), 4)


def build_pairwise_agreement_matrix(
    predictions_by_scenario: List[Dict[str, Dict[str, Any]]]
) -> Dict[str, float]:
    """Pairwise agreement between stage pairs (stage1 vs stage3, stage3 vs stage4, etc.)."""
    pairs = [("stage1", "stage3"), ("stage3", "stage4"), ("stage1", "stage4")]
    agreements = {f"{p1}_vs_{p2}": 0.0 for p1, p2 in pairs}
    counts = {f"{p1}_vs_{p2}": 0 for p1, p2 in pairs}
    
    for sc in predictions_by_scenario:
        for p1, p2 in pairs:
            if p1 in sc and p2 in sc:
                v1 = sc[p1].get("risk_category") or sc[p1].get("urgency")
                v2 = sc[p2].get("risk_category") or sc[p2].get("urgency")
                if v1 and v2:
                    counts[f"{p1}_vs_{p2}"] += 1
                    if v1 == v2:
                        agreements[f"{p1}_vs_{p2}"] += 1.0

    result = {}
    for pair_key, total in counts.items():
        result[pair_key] = round(agreements[pair_key] / max(1, total), 4)
    return result
