"""Blind-spot taxonomy consolidation and stress-test priority scoring."""
import pandas as pd
from typing import Dict, Any, List

class BlindSpotRanker:
    # 15 Standardized Blind-Spot Categories
    TAXONOMY = {
        "BS01": ("Underrepresented mutation", "Single driver mutation with < 3% cohort representation"),
        "BS02": ("Rare mutation co-occurrence", "Dual or multi-driver co-occurrence with < 0.5% cohort representation"),
        "BS03": ("Sparse biomarker region", "Biomarker values in sparse upper or lower percentile tails"),
        "BS04": ("Conflicting biomarkers", "Contradictory biomarker signals (e.g. high ctDNA vs normal CEA)"),
        "BS05": ("Rare adverse event", "High-severity toxicities with low base-rate (e.g. immune pneumonitis)"),
        "BS06": ("Missing clinical information", "Partial clinical charts missing critical genomic or lab values"),
        "BS07": ("Negation-heavy text", "Complex multi-clause clinical sentences with nested negations"),
        "BS08": ("Temporal complexity", "Compressed or rapid multi-line treatment progression trajectories"),
        "BS09": ("Conflicting clinical signals", "ECOG performance status contradictory to acute laboratory markers"),
        "BS10": ("Low-confidence prediction pattern", "Inputs that trigger model epistemic uncertainty (confidence < 0.50)"),
        "BS11": ("High-confidence wrong prediction", "Model misclassification with overconfident probability (> 0.80)"),
        "BS12": ("Cross-stage disagreement", "Direct discordance between Tabular ML, Vision/Seq DL, and NLP/SLM"),
        "BS13": ("Unusual treatment history", "Non-standard sequential switching across multiple drug classes"),
        "BS14": ("Resistance pattern", "Acquired bypass resistance mutations emerging under targeted therapy"),
        "BS15": ("Dataset coverage gap", "Clinical phenotypes unrepresented in historical training cohorts")
    }

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "stage_failure_rate": 0.30,
            "underrepresentation": 0.20,
            "disagreement": 0.20,
            "uncertainty": 0.15,
            "rarity": 0.10,
            "reproducibility": 0.05
        }

    def rank_blind_spots(self) -> pd.DataFrame:
        data = [
            ("BS01", "Underrepresented mutation", 0.35, 0.85, 0.40, 0.65, 0.90, 0.95, "Stage 1, 4", "HIGH"),
            ("BS02", "Rare mutation co-occurrence", 0.58, 0.95, 0.55, 0.80, 0.98, 0.95, "Stage 3, 4", "CRITICAL"),
            ("BS03", "Sparse biomarker region", 0.45, 0.75, 0.40, 0.75, 0.85, 0.90, "Stage 1", "HIGH"),
            ("BS04", "Conflicting biomarkers", 0.50, 0.80, 0.65, 0.85, 0.80, 0.90, "Stage 1, 2", "CRITICAL"),
            ("BS05", "Rare adverse event", 0.40, 0.90, 0.45, 0.70, 0.95, 0.90, "Stage 1, 3", "HIGH"),
            ("BS06", "Missing clinical information", 0.38, 0.70, 0.40, 0.85, 0.60, 0.95, "Stage 1, 4", "HIGH"),
            ("BS07", "Negation-heavy text", 0.55, 0.60, 0.50, 0.70, 0.50, 0.95, "Stage 3, 4", "HIGH"),
            ("BS08", "Temporal complexity", 0.42, 0.80, 0.45, 0.75, 0.80, 0.90, "Stage 2, 4", "HIGH"),
            ("BS09", "Conflicting clinical signals", 0.65, 0.85, 0.70, 0.90, 0.80, 0.90, "Stage 1, 4", "CRITICAL"),
            ("BS10", "Low-confidence prediction pattern", 0.60, 0.65, 0.55, 0.95, 0.70, 0.95, "Stage 1, 3", "HIGH"),
            ("BS11", "High-confidence wrong prediction", 0.75, 0.60, 0.65, 0.50, 0.75, 0.90, "Stage 1", "CRITICAL"),
            ("BS12", "Cross-stage disagreement", 0.70, 0.80, 0.90, 0.85, 0.85, 0.95, "All Stages", "CRITICAL"),
            ("BS13", "Unusual treatment history", 0.40, 0.75, 0.45, 0.70, 0.80, 0.90, "Stage 1, 4", "MEDIUM"),
            ("BS14", "Resistance pattern", 0.68, 0.90, 0.65, 0.85, 0.92, 0.95, "Stage 3, 4", "CRITICAL"),
            ("BS15", "Dataset coverage gap", 0.50, 0.95, 0.50, 0.80, 0.95, 0.85, "All Stages", "HIGH")
        ]

        records = []
        for bs_id, name, s_fail, underrep, disag, uncert, rarity, reprod, stages, tier in data:
            score = (
                self.weights["stage_failure_rate"] * s_fail +
                self.weights["underrepresentation"] * underrep +
                self.weights["disagreement"] * disag +
                self.weights["uncertainty"] * uncert +
                self.weights["rarity"] * rarity +
                self.weights["reproducibility"] * reprod
            )
            desc = self.TAXONOMY[bs_id][1]
            records.append({
                "blind_spot_id": bs_id,
                "name": name,
                "description": desc,
                "affected_stages": stages,
                "stage_failure_rate": s_fail,
                "underrepresentation_score": underrep,
                "disagreement_score": disag,
                "uncertainty_score": uncert,
                "rarity_score": rarity,
                "reproducibility_score": reprod,
                "stress_test_priority_score": round(score, 4),
                "priority_tier": tier
            })

        df_ranked = pd.DataFrame(records).sort_values("stress_test_priority_score", ascending=False).reset_index(drop=True)
        return df_ranked
