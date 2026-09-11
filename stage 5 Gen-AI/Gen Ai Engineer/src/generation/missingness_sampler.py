"""Sampler for intentional multi-field clinical record missingness."""
import random
from typing import List, Dict, Any


class MissingnessSampler:
    def __init__(self):
        self.common_missing_pool = [
            "smoking_pack_years", "family_history", "baseline_ecog",
            "her2_ihc_score", "ki67_index", "cea_level", "ctdna_level"
        ]

    def sample_missing_fields(self, scenario: Dict[str, Any], rng: random.Random) -> List[str]:
        sid = scenario.get("scenario_id", "")
        cat = scenario.get("category", "")

        if "SPARSE_MATRIX" in cat or sid == "PROMPT-R07":
            # Explicit high-missingness transfer envelope (>60% missing)
            return [
                "tumor_size_cm", "cea_level", "smoking_pack_years",
                "mutation_profile", "er_status", "pr_status", "her2_status"
            ]
        
        # Standard scenario: mild realistic missingness (0 to 2 unessential fields)
        n = rng.choice([0, 1, 2])
        if n > 0:
            return rng.sample(self.common_missing_pool, n)
        return []