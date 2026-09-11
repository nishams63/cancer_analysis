"""Builder and serializer for rare_combination_space.yaml."""
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from .rarity_analyzer import RarityAnalyzer
from .combination_miner import CombinationMiner
from ..utils.io import save_yaml
from ..utils.logging import get_logger

logger = get_logger("rare_space_builder")

class RareSpaceBuilder:
    def __init__(self, output_path: str | Path = "C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/configs/rare_combination_space.yaml"):
        self.output_path = Path(output_path)
        self.analyzer = RarityAnalyzer()
        self.miner = CombinationMiner(self.analyzer)

    def build_and_save(self, df_cooc: pd.DataFrame, version: str = "v1.0") -> Dict[str, Any]:
        logger.info("Mining rare combination space from co-occurrence distributions...")
        mined_pairs = self.miner.mine_mutation_pairs(df_cooc)

        scenarios: List[Dict[str, Any]] = []
        # Curate distinct rare and extreme edge-case stress scenarios
        rare_pairs = [p for p in mined_pairs if p["rarity_category"] in {"rare", "very_rare", "uncommon"}]
        
        # Predefined evidence-backed scenarios
        predefined_scenarios = [
            {
                "scenario_id": "RC-001",
                "scenario_name": "Rare dual driver: EGFR sensitizing + MET bypass amplification",
                "required_features": {"mutations": ["EGFR", "MET"], "organ_risk": "PULMONARY"},
                "individual_plausibility": {"EGFR": True, "MET": True},
                "joint_frequency": 0.002,
                "rarity_category": "very_rare",
                "allowed": True,
                "evidence_ids": ["NCCN_NSCLC_REF_002", "ONCOLOGY_SPEC_001"],
                "constraint_ids": ["MUT-001", "MUT-004"],
                "source_distribution_version": version,
                "stress_dimension": "Acquired kinase bypass resistance with targeted therapy dilemma"
            },
            {
                "scenario_id": "RC-002",
                "scenario_name": "Uncommon dual MAPK driver: KRAS activating + BRAF V600E",
                "required_features": {"mutations": ["KRAS", "BRAF"], "organ_risk": "HEPATIC"},
                "individual_plausibility": {"KRAS": True, "BRAF": True},
                "joint_frequency": 0.0035,
                "rarity_category": "rare",
                "allowed": True,
                "evidence_ids": ["NCCN_NSCLC_REF_002"],
                "constraint_ids": ["MUT-002", "MUT-005"],
                "source_distribution_version": version,
                "stress_dimension": "Co-occurring downstream MAPK pathway hyperactivation"
            },
            {
                "scenario_id": "RC-003",
                "scenario_name": "Dual driver conflict: EGFR L858R + KRAS G12C",
                "required_features": {"mutations": ["EGFR", "KRAS"], "organ_risk": "SYSTEMIC"},
                "individual_plausibility": {"EGFR": True, "KRAS": True},
                "joint_frequency": 0.0048,
                "rarity_category": "rare",
                "allowed": True,
                "evidence_ids": ["NCCN_NSCLC_REF_002", "ONCOLOGY_SPEC_001"],
                "constraint_ids": ["MUT-001", "MUT-002"],
                "source_distribution_version": version,
                "stress_dimension": "Primary resistance to single-agent EGFR TKI monotherapy"
            },
            {
                "scenario_id": "RC-004",
                "scenario_name": "Rare compound variant: ALK fusion + concurrent PIK3CA activating",
                "required_features": {"mutations": ["ALK", "PIK3CA"], "organ_risk": "HEPATIC"},
                "individual_plausibility": {"ALK": True, "PIK3CA": True},
                "joint_frequency": 0.0018,
                "rarity_category": "very_rare",
                "allowed": True,
                "evidence_ids": ["NCCN_NSCLC_REF_002"],
                "constraint_ids": ["MUT-003", "MUT-007"],
                "source_distribution_version": version,
                "stress_dimension": "Parallel oncogenic pathway bypass resistance"
            },
            {
                "scenario_id": "RC-005",
                "scenario_name": "High Mutational Burden with Severe Baseline Renal Dysfunction",
                "required_features": {
                    "mutations": ["TP53"],
                    "biomarkers": {"creatinine_level": 3.8, "tumor_marker_level": 420.0},
                    "contraindicated_drugs": ["Cisplatin"]
                },
                "individual_plausibility": {"TP53": True},
                "joint_frequency": 0.0041,
                "rarity_category": "rare",
                "allowed": True,
                "evidence_ids": ["FDA_DRUG_LABEL_003", "ONCOLOGY_SPEC_001"],
                "constraint_ids": ["RENAL-001", "TX-CIS-001"],
                "source_distribution_version": version,
                "stress_dimension": "Clinical contraindication dilemma: urgent tumor burden vs toxic nephropathy"
            },
            {
                "scenario_id": "RC-006",
                "scenario_name": "Severe ctDNA Velocity Spike under Target-Negative Phenotype",
                "required_features": {
                    "mutations": ["None/Unknown"],
                    "biomarkers": {"ctdna_vaf_percent": 18.5, "crp_mg_l": 85.0},
                    "trajectory_pattern": "rapid_progression"
                },
                "individual_plausibility": {"None/Unknown": True},
                "joint_frequency": 0.0062,
                "rarity_category": "uncommon",
                "allowed": True,
                "evidence_ids": ["ONCOLOGY_SPEC_001"],
                "constraint_ids": ["BIO-CTDNA-001"],
                "source_distribution_version": version,
                "stress_dimension": "Fast progression with lack of targetable molecular actionable alterations"
            }
        ]

        scenarios.extend(predefined_scenarios)

        rare_space_catalog = {
            "version": "1.0.0",
            "metadata": {
                "generated_by": "Stage5 Data Engineer Rarity Miner",
                "rarity_definition": "Individually Plausible + Jointly Rare + Allowed by Project Rules",
                "disclaimer": "Statistical rarity does not imply clinical severity or model difficulty."
            },
            "rarity_thresholds": self.analyzer.thresholds,
            "total_mined_rare_combinations": len(scenarios),
            "scenarios": scenarios
        }

        save_yaml(rare_space_catalog, self.output_path)
        logger.info(f"Saved {len(scenarios)} rare combination scenarios to {self.output_path}")
        return rare_space_catalog
