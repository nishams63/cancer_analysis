"""Sampler for rare mutation co-occurrences and resistance profiles."""
import random
from typing import List, Dict, Any, Optional


class RaritySampler:
    """Samples rare combinations from rare_combination_space and scenario specifications."""
    def __init__(self, rare_space_config: Dict[str, Any] = None):
        self.rare_space = rare_space_config or {}

    def sample_mutations_for_scenario(self, scenario: Dict[str, Any], rng: random.Random) -> List[str]:
        """Extract or sample mutations fulfilling scenario required entities."""
        mutations = []
        req_entities = scenario.get("required_entities", [])
        
        for req in req_entities:
            etype = req.get("entity_type")
            if etype == "mutation":
                allowed = req.get("allowed_values", [])
                if allowed:
                    chosen = rng.choice(allowed)
                    # Split if multi-mutation string like "EGFR L858R + T790M"
                    if " + " in chosen:
                        mutations.extend(chosen.split(" + "))
                    else:
                        mutations.append(chosen)
                else:
                    mutations.append(req.get("name", "Unknown_Mutation"))

        # Fallback if no specific mutations in required_entities
        if not mutations:
            cat = scenario.get("category", "")
            if "EGFR" in cat or "DUAL_MUTATION" in cat:
                mutations = ["EGFR T790M", "MET Amplification"]
            elif "QUADRUPLE" in cat:
                mutations = ["KRAS G12C", "TP53 R273H", "STK11 loss", "KEAP1 truncating"]
            else:
                mutations = ["TP53", "KRAS G12D"]

        return list(dict.fromkeys(mutations))  # Deduplicate preserving order