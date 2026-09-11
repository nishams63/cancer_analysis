"""Stress-test sensitivity drop and discordance metrics."""
from typing import List, Dict, Any


def calculate_sensitivity_drop(standard_sens: float, stress_sens: float) -> float:
    """Calculate absolute sensitivity degradation: Delta Sens = Sens_standard - Sens_stress."""
    return round(max(0.0, standard_sens - stress_sens), 4)


def calculate_discordance_rate(eval_results: List[Dict[str, Any]]) -> float:
    if not eval_results:
        return 0.0
    exposed = sum(1 for r in eval_results if r.get("is_vulnerability_exposed"))
    return round(exposed / len(eval_results), 4)
