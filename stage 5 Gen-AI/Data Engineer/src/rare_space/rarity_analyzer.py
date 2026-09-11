"""Rarity categorization and threshold evaluation."""
from typing import Dict, Any

class RarityAnalyzer:
    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or {
            "common": 0.10,
            "uncommon": 0.03,
            "rare": 0.005,
            "very_rare": 0.001
        }

    def categorize_frequency(self, freq: float) -> str:
        if freq >= self.thresholds["common"]:
            return "common"
        elif freq >= self.thresholds["uncommon"]:
            return "uncommon"
        elif freq >= self.thresholds["rare"]:
            return "rare"
        else:
            return "very_rare"

    def is_rare(self, freq: float) -> bool:
        return freq < self.thresholds["uncommon"]
