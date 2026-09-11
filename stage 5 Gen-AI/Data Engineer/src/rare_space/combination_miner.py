"""Mining of candidate multi-feature combinations satisfying plausibility and rarity."""
import pandas as pd
from typing import List, Dict, Any
from .rarity_analyzer import RarityAnalyzer

class CombinationMiner:
    def __init__(self, analyzer: RarityAnalyzer):
        self.analyzer = analyzer

    def mine_mutation_pairs(self, df_cooc: pd.DataFrame) -> List[Dict[str, Any]]:
        candidates = []
        for _, row in df_cooc.iterrows():
            freq = float(row["cooccurrence_frequency"])
            cat = self.analyzer.categorize_frequency(freq)
            # Both mutations must be individually plausible (> 0 marginal)
            indiv_plausible = (row["individual_frequency_A"] > 0) and (row["individual_frequency_B"] > 0)
            candidates.append({
                "pair": [str(row["mutation_A"]), str(row["mutation_B"])],
                "joint_frequency": freq,
                "rarity_category": cat,
                "individual_plausibility": {
                    str(row["mutation_A"]): bool(row["individual_frequency_A"] > 0),
                    str(row["mutation_B"]): bool(row["individual_frequency_B"] > 0)
                },
                "allowed": indiv_plausible and (row["mutation_A"] != row["mutation_B"])
            })
        return candidates
