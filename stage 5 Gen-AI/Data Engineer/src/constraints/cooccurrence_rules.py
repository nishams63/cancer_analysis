"""Genomic mutation co-occurrence and exclusivity constraint rules."""
from typing import Dict, Any

class CooccurrenceRulesBuilder:
    def build(self) -> Dict[str, Any]:
        return {
            "allowed_driver_mutations": [
                "EGFR", "KRAS", "TP53", "ALK", "MET", "BRAF", "PIK3CA", "HER2", "ROS1", "RET"
            ],
            "common_cooccurrences": [
                {"pair": ["KRAS", "TP53"], "evidence_status": "empirically_frequent", "frequency_range": [0.35, 0.45]},
                {"pair": ["EGFR", "TP53"], "evidence_status": "empirically_frequent", "frequency_range": [0.30, 0.40]}
            ],
            "rare_cooccurrences": [
                {"pair": ["EGFR", "MET"], "evidence_status": "bypass_resistance_mechanism", "joint_frequency_bound": 0.02},
                {"pair": ["KRAS", "BRAF"], "evidence_status": "rare_dual_mapk", "joint_frequency_bound": 0.01},
                {"pair": ["EGFR", "KRAS"], "evidence_status": "rare_dual_driver", "joint_frequency_bound": 0.015}
            ],
            "forbidden_contradictions": [
                {"rule_id": "MUT-EXCL-001", "pair": ["ALK", "ROS1"], "reason": "Dual kinase fusions mutually exclusive in treatment-naive presentation"},
                {"rule_id": "MUT-SYNTAX-002", "condition": "mutation_name not in allowed_driver_mutations and mutation_name != 'None/Unknown'"}
            ],
            "unknown_handling": {
                "policy": "allow_unmutated",
                "canonical_representation": "None/Unknown"
            }
        }
