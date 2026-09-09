"""
Risk Distribution and Stratified Information Loss Analysis Module for Stage 4 EDA.
Quantifies class balance (Low, Moderate, High), Shannon entropy, imbalance ratios,
and audits stratified retention and token metrics across clinical risk tiers.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import entropy


def map_clinical_risk_tier(df: pd.DataFrame, stage3_df: Optional[pd.DataFrame] = None) -> pd.Series:
    """
    Assigns each record to a canonical Risk Tier: 'Low', 'Moderate', or 'High'.
    Maps Stage 3 urgency_level (LOW -> Low, MEDIUM -> Moderate, HIGH/CRITICAL -> High)
    or infers from target_action text if Stage 3 reference is unlinked.
    """
    if stage3_df is not None and "urgency_level" in stage3_df.columns:
        urgency_map = dict(zip(stage3_df["document_id"], stage3_df["urgency_level"]))
        mapped = df["note_id"].map(urgency_map)
        if mapped.notna().sum() == len(df):
            tier_map = {
                "LOW": "Low",
                "MEDIUM": "Moderate",
                "HIGH": "High",
                "CRITICAL": "High"
            }
            return mapped.map(tier_map).fillna("Low")

    # Fallback to deterministic mapping via action semantics
    tiers = []
    for action in df["target_action"].fillna(""):
        act_lower = str(action).lower()
        if "hold or reduce" in act_lower:
            tiers.append("High")
        elif "monitor" in act_lower and "closely" in act_lower:
            tiers.append("Moderate")
        else:
            tiers.append("Low")
    return pd.Series(tiers, index=df.index)


class RiskAnalyzer:
    """Audits risk class balance and checks for risk-stratified information loss."""

    def analyze_risk_distribution(self, risk_series: pd.Series) -> Dict[str, Any]:
        """Calculates distribution statistics, entropy, and imbalance ratios."""
        counts = risk_series.value_counts().to_dict()
        total = len(risk_series)

        # Ensure Low, Moderate, High are all present
        for tier in ["Low", "Moderate", "High"]:
            if tier not in counts:
                counts[tier] = 0

        percentages = {tier: float(round((cnt / total) * 100.0, 2)) for tier, cnt in counts.items()}
        probs = [cnt / total for cnt in counts.values() if cnt > 0]
        shannon_entropy = float(round(float(entropy(probs, base=2)), 4))

        max_count = max(counts.values())
        min_count = min([c for c in counts.values() if c > 0] or [1])
        imbalance_ratio = float(round(max_count / min_count, 2))
        minority_pct = float(round((min_count / total) * 100.0, 2))

        return {
            "counts": counts,
            "percentages": percentages,
            "shannon_entropy": shannon_entropy,
            "imbalance_ratio": imbalance_ratio,
            "minority_class_percentage": minority_pct,
            "majority_class": max(counts, key=counts.get),
            "minority_class": min(counts, key=counts.get)
        }

    def analyze_stratified_information_loss(
        self,
        df: pd.DataFrame,
        risk_series: pd.Series,
        source_tokens: np.ndarray,
        target_tokens: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculates information retention across Low, Moderate, and High risk tiers:
        - average source tokens & target tokens
        - average source entities & target entities
        - entity retention rate
        """
        df_eval = df.copy()
        df_eval["risk_tier"] = risk_series
        df_eval["source_tokens"] = source_tokens
        df_eval["target_tokens"] = target_tokens

        # Helper to count entities
        def count_row_entities(row, cols):
            cnt = 0
            for c in cols:
                ents = row.get(c, [])
                if isinstance(ents, (list, np.ndarray)):
                    cnt += sum(1 for e in ents if str(e).lower() not in ("none/unknown", "none", ""))
            return cnt

        entity_cols = ["ner_genes", "ner_drugs", "ner_dosages", "ner_adverse_events"]
        df_eval["source_entities"] = df_eval.apply(lambda r: count_row_entities(r, entity_cols), axis=1)

        target_corpus = (
            df_eval["target_risk"].fillna("")
            + " "
            + df_eval["target_key_finding"].fillna("")
            + " "
            + df_eval["target_action"].fillna("")
        ).str.lower()

        def count_preserved_entities(row, target_str):
            preserved = 0
            for c in entity_cols:
                ents = row.get(c, [])
                if isinstance(ents, (list, np.ndarray)):
                    for e in ents:
                        e_str = str(e).strip().lower()
                        if e_str and e_str not in ("none/unknown", "none", "") and e_str in target_str:
                            preserved += 1
            return preserved

        df_eval["preserved_entities"] = [
            count_preserved_entities(row, target_corpus.iloc[i])
            for i, row in df_eval.iterrows()
        ]

        stratified_report = {}
        for tier in ["Low", "Moderate", "High"]:
            subset = df_eval[df_eval["risk_tier"] == tier]
            if len(subset) == 0:
                continue

            src_ents_sum = subset["source_entities"].sum()
            pres_ents_sum = subset["preserved_entities"].sum()
            ret_rate = float(round(pres_ents_sum / src_ents_sum, 4)) if src_ents_sum > 0 else 1.0

            stratified_report[tier] = {
                "record_count": int(len(subset)),
                "average_source_tokens": float(round(float(subset["source_tokens"].mean()), 1)),
                "average_target_tokens": float(round(float(subset["target_tokens"].mean()), 1)),
                "average_source_entities": float(round(float(subset["source_entities"].mean()), 2)),
                "average_target_entities": float(round(float(subset["preserved_entities"].mean()), 2)),
                "entity_retention_rate": ret_rate
            }

        return stratified_report
