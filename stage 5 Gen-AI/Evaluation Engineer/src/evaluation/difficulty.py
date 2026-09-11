"""System Stress-Test Difficulty Score Calculator."""
from typing import Dict, Any


class DifficultyCalculator:
    """Calculates SYSTEM STRESS-TEST DIFFICULTY SCORE (0-100)."""

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "stage_failure": 0.40,
            "disagreement": 0.20,
            "uncertainty": 0.20,
            "instability": 0.20
        }

    def calculate_difficulty(
        self,
        failure_count: int,
        has_disagreement: bool,
        confidence_spread: float,
        has_instability: bool = False
    ) -> Dict[str, Any]:
        # 1. Stage failure component (0 to 1.0): 4 failures = 1.0
        f_score = min(1.0, failure_count / 3.0)

        # 2. Disagreement component (0 or 1.0)
        d_score = 1.0 if has_disagreement else 0.0

        # 3. Uncertainty component (confidence spread up to 0.40 = 1.0)
        u_score = min(1.0, confidence_spread / 0.30)

        # 4. Instability component
        i_score = 1.0 if has_instability else 0.2

        w = self.weights
        total_difficulty = (
            w.get("stage_failure", 0.40) * f_score +
            w.get("disagreement", 0.20) * d_score +
            w.get("uncertainty", 0.20) * u_score +
            w.get("instability", 0.20) * i_score
        ) * 100.0

        total_difficulty = round(min(100.0, max(0.0, total_difficulty)), 2)

        # Tier classification
        if total_difficulty >= 75.0:
            tier = "EXTREME"
        elif total_difficulty >= 50.0:
            tier = "HARD"
        elif total_difficulty >= 25.0:
            tier = "MODERATE"
        else:
            tier = "LOW"

        return {
            "system_stress_test_difficulty_score": total_difficulty,
            "difficulty_tier": tier,
            "components": {
                "stage_failure_score": round(f_score * 100, 1),
                "disagreement_score": round(d_score * 100, 1),
                "uncertainty_score": round(u_score * 100, 1),
                "instability_score": round(i_score * 100, 1)
            }
        }
