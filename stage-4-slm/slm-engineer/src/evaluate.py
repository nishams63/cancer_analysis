"""
Clinical Evaluation Engine for Stage 5 SLM Engineering.
Computes empirical performance metrics across:
- Categorical Risk Macro-F1 & Accuracy
- Upstream Stage 3 Entity Retention Rate
- Hallucination Rate
- Negation Preservation Rate & Negation Flips
- Schema Format Compliance
- Composite Selection Score
"""

import re
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from .utils import setup_logger

logger = setup_logger("evaluate")


class ClinicalEvaluator:
    """Evaluates SLM generated responses against reference ground truth and entities."""

    VALID_RISK_TIERS = ["Low", "Moderate", "High"]

    @staticmethod
    def evaluate_risk_classification(
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Calculates Categorical Accuracy and Macro-F1 on Risk tiers."""
        clean_preds = [p if p in ClinicalEvaluator.VALID_RISK_TIERS else "Invalid" for p in predictions]
        clean_refs = [r if r in ClinicalEvaluator.VALID_RISK_TIERS else "Invalid" for r in references]

        acc = float(accuracy_score(clean_refs, clean_preds))
        macro_f1 = float(f1_score(clean_refs, clean_preds, average="macro", zero_division=0))

        return {
            "risk_accuracy": round(acc, 4),
            "risk_macro_f1": round(macro_f1, 4)
        }

    @staticmethod
    def evaluate_entity_retention(
        generated_texts: List[str],
        reference_entities_list: List[Dict[str, List[str]]]
    ) -> Dict[str, float]:
        """
        Measures the fraction of upstream Stage 3 clinical entities
        (genes, drugs, dosages, adverse events) preserved in the output.
        """
        total_entities = 0
        retained_entities = 0

        for gen_text, ref_ents in zip(generated_texts, reference_entities_list):
            gen_lower = gen_text.lower()
            all_target_ents = []
            for ent_type in ["genes", "drugs", "dosages", "adverse_events"]:
                ents = ref_ents.get(f"ner_{ent_type}", [])
                if isinstance(ents, (list, np.ndarray)):
                    all_target_ents.extend([str(e).lower().strip() for e in ents if e])

            for ent in all_target_ents:
                total_entities += 1
                if ent in gen_lower:
                    retained_entities += 1

        retention_rate = (retained_entities / total_entities) if total_entities > 0 else 1.0
        return {
            "total_reference_entities": total_entities,
            "retained_entities": retained_entities,
            "entity_retention_rate": round(retention_rate, 4)
        }

    @staticmethod
    def evaluate_hallucination_rate(
        generated_texts: List[str],
        source_notes: List[str]
    ) -> Dict[str, float]:
        """
        Detects ungrounded antineoplastics or acute adverse events
        mentioned in the output that are absent from the source note.
        """
        common_antineoplastics = [
            "osimertinib", "docetaxel", "pembrolizumab", "trastuzumab", "cisplatin",
            "carboplatin", "paclitaxel", "erlotinib", "gefitinib", "doxorubicin",
            "methotrexate", "nivolumab", "ipilimumab", "gemcitabine"
        ]

        hallucination_count = 0
        total_eval_samples = len(generated_texts)

        for gen_text, note in zip(generated_texts, source_notes):
            gen_lower = gen_text.lower()
            note_lower = note.lower()

            has_hallucination = False
            for drug in common_antineoplastics:
                if drug in gen_lower and drug not in note_lower:
                    has_hallucination = True
                    break

            if has_hallucination:
                hallucination_count += 1

        hallucination_rate = (hallucination_count / total_eval_samples) if total_eval_samples > 0 else 0.0
        return {
            "hallucination_count": hallucination_count,
            "hallucination_rate": round(hallucination_rate, 4)
        }

    @staticmethod
    def evaluate_negation_preservation(
        generated_texts: List[str],
        source_notes: List[str]
    ) -> Dict[str, float]:
        """
        Detects critical negation polarity flips (e.g., source note explicitly
        denies toxicities, but generation attributes high risk / severe toxicities).
        """
        negation_cues = [
            "no acute adverse", "denies adverse", "denies toxicities", "denies any",
            "no toxicities", "without acute toxicity", "free of treatment-limiting",
            "zero adverse symptoms", "absence of acute", "tolerating well with no"
        ]

        flips = 0
        negated_samples = 0

        for gen_text, note in zip(generated_texts, source_notes):
            note_lower = note.lower()
            is_negated = any(phrase in note_lower for phrase in negation_cues)

            if is_negated:
                negated_samples += 1
                # If note has clear negation, model should not assign High risk or claim severe toxicities
                if "risk: high" in gen_text.lower() or "severe treatment-related" in gen_text.lower():
                    flips += 1

        flip_rate = (flips / negated_samples) if negated_samples > 0 else 0.0
        preservation_rate = 1.0 - flip_rate

        return {
            "negated_samples_count": negated_samples,
            "negation_flips": flips,
            "negation_flip_rate": round(flip_rate, 4),
            "negation_preservation_rate": round(preservation_rate, 4)
        }

    @staticmethod
    def evaluate_format_compliance(
        parsed_outputs: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Measures the proportion of outputs adhering strictly to the Risk/Finding/Action schema."""
        valid_count = sum(1 for p in parsed_outputs if p.get("is_valid_format", False))
        compliance_rate = (valid_count / len(parsed_outputs)) if parsed_outputs else 0.0
        return {
            "valid_format_count": valid_count,
            "total_evaluated": len(parsed_outputs),
            "format_compliance_rate": round(compliance_rate, 4)
        }

    @staticmethod
    def compute_composite_selection_score(metrics: Dict[str, float]) -> float:
        """
        Calculates official Stage 5 composite selection score:
        0.30 x Macro-F1 + 0.25 x Retention + 0.25 x NegationPres + 0.10 x FormatComp - 0.10 x Hallucination
        """
        f1 = metrics.get("risk_macro_f1", 0.0)
        retention = metrics.get("entity_retention_rate", 0.0)
        negation = metrics.get("negation_preservation_rate", 0.0)
        compliance = metrics.get("format_compliance_rate", 0.0)
        hallucination = metrics.get("hallucination_rate", 0.0)

        score = (
            0.30 * f1 +
            0.25 * retention +
            0.25 * negation +
            0.10 * compliance -
            0.10 * hallucination
        )
        return round(float(score), 4)
