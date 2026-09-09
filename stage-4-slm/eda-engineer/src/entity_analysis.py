"""
Entity Density and Retention Analysis Module for Stage 4 EDA.
Measures reference entity frequencies, density per note, and retention fidelity
across entity classes (Gene, Drug, Dosage, Adverse Event) and risk tiers.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class EntityAnalyzer:
    """Performs entity density and target retention audits."""

    ENTITY_COLS = {
        "gene": "ner_genes",
        "drug": "ner_drugs",
        "dosage": "ner_dosages",
        "adverse_event": "ner_adverse_events"
    }

    @staticmethod
    def _clean_entities(raw_entities) -> List[str]:
        """Cleans and filters uninformative or placeholder entities."""
        cleaned = []
        if isinstance(raw_entities, (list, np.ndarray)):
            for e in raw_entities:
                e_str = str(e).strip()
                if e_str and e_str.lower() not in ("none/unknown", "none", "unknown", ""):
                    cleaned.append(e_str)
        return cleaned

    def analyze_entity_density(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates per-note entity counts across Gene, Drug, Dosage, and Adverse Event.
        """
        density_by_category = {}
        total_entities_per_note = np.zeros(len(df), dtype=int)

        for cat, col in self.ENTITY_COLS.items():
            if col in df.columns:
                counts = df[col].apply(lambda x: len(self._clean_entities(x))).values
            else:
                counts = np.zeros(len(df), dtype=int)
            total_entities_per_note += counts
            density_by_category[cat] = self._compute_density_stats(counts)

        density_by_category["overall"] = self._compute_density_stats(total_entities_per_note)

        return {
            "density_metrics": density_by_category,
            "density_arrays": {
                cat: df[col].apply(lambda x: len(self._clean_entities(x))).values if col in df.columns else np.zeros(len(df), dtype=int)
                for cat, col in self.ENTITY_COLS.items()
            },
            "overall_array": total_entities_per_note
        }

    @staticmethod
    def _compute_density_stats(arr: np.ndarray) -> Dict[str, float]:
        if len(arr) == 0:
            return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0}
        return {
            "mean": float(round(float(np.mean(arr)), 2)),
            "median": float(round(float(np.median(arr)), 2)),
            "p90": float(round(float(np.percentile(arr, 90)), 2)),
            "p95": float(round(float(np.percentile(arr, 95)), 2)),
            "max": int(np.max(arr))
        }

    def analyze_entity_retention(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates retention of reference entities in target outputs across categories
        and stratified by risk level.
        """
        # Combine target text into one searchable string per row
        target_corpus = (
            df["target_risk"].fillna("")
            + " "
            + df["target_key_finding"].fillna("")
            + " "
            + df["target_action"].fillna("")
        ).str.lower()

        category_retention = {}
        for cat, col in self.ENTITY_COLS.items():
            source_count = 0
            preserved_count = 0

            if col in df.columns:
                for idx, row in df.iterrows():
                    ents = self._clean_entities(row[col])
                    t_text = target_corpus.iloc[idx]
                    for ent in ents:
                        source_count += 1
                        # Case-insensitive entity match in target
                        if ent.lower() in t_text:
                            preserved_count += 1

            missing_count = source_count - preserved_count
            ret_rate = float(round((preserved_count / source_count), 4)) if source_count > 0 else 1.0

            category_retention[cat] = {
                "source_count": source_count,
                "preserved_count": preserved_count,
                "missing_count": missing_count,
                "retention_rate": ret_rate
            }

        total_source = sum(c["source_count"] for c in category_retention.values())
        total_preserved = sum(c["preserved_count"] for c in category_retention.values())
        overall_retention = float(round(total_preserved / total_source, 4)) if total_source > 0 else 1.0

        return {
            "overall_retention_rate": overall_retention,
            "total_source_entities": total_source,
            "total_preserved_entities": total_preserved,
            "categories": category_retention
        }
