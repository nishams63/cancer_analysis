"""
Clinical, Structural, and Safety Evaluation Metrics Module for Stage 5 SLM.
Computes format compliance, risk macro-F1, reference entity retention,
hallucination penalties, and clinical negation flip rates per Sections 19-26.
"""

import re
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from prompt_template import ClinicalPromptTemplate


class ClinicalEvaluator:
    """Evaluates generated SLM responses against reference clinical criteria."""

    KNOWN_DRUGS = [
        "cisplatin", "carboplatin", "paclitaxel", "pemetrexed", "erlotinib",
        "gefitinib", "osimertinib", "alectinib", "crizotinib", "pembrolizumab",
        "durvalumab", "docetaxel", "dabrafenib", "trametinib", "bevacizumab",
        "trastuzumab", "fluorouracil", "5-fu"
    ]

    KNOWN_GENES = [
        "egfr", "kras", "alk", "braf", "pd-l1", "her2", "ros1", "ret", "tp53", "met"
    ]

    def __init__(self, prompt_template: Optional[ClinicalPromptTemplate] = None):
        self.prompt_template = prompt_template or ClinicalPromptTemplate()

    def evaluate_batch(
        self,
        predictions: List[str],
        targets: List[str],
        expected_risks: List[str],
        clinical_notes: List[str],
        reference_entities_list: List[Dict[str, List[str]]],
        patient_ids: Optional[List[str]] = None,
        note_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Runs complete evaluation across all structural, clinical, and safety dimensions.
        """
        total = len(predictions)
        if total == 0:
            return {"error": "Empty evaluation batch"}

        # 1. Structural Format Compliance (Section 19)
        parsed_preds = [self.prompt_template.parse_slm_output(p) for p in predictions]
        compliant_count = sum(1 for p in parsed_preds if p["is_compliant"])
        format_compliance = float(round(compliant_count / total, 4))
        missing_risk = sum(1 for p in parsed_preds if "Risk" in p["missing_fields"]) / total
        missing_kf = sum(1 for p in parsed_preds if "Key Finding" in p["missing_fields"]) / total
        missing_action = sum(1 for p in parsed_preds if "Action" in p["missing_fields"]) / total

        # 2. Risk Metrics (Section 20)
        pred_risks = [self.prompt_template._extract_risk_tier(p["risk"] or "") for p in parsed_preds]
        risk_classes = ["Low", "Moderate", "High"]

        acc = float(round(accuracy_score(expected_risks, pred_risks), 4))
        prec, rec, f1, _ = precision_recall_fscore_support(
            expected_risks, pred_risks, labels=risk_classes, average="macro", zero_division=0
        )
        per_cls_p, per_cls_r, per_cls_f1, _ = precision_recall_fscore_support(
            expected_risks, pred_risks, labels=risk_classes, average=None, zero_division=0
        )
        conf_mat = confusion_matrix(expected_risks, pred_risks, labels=risk_classes).tolist()

        per_class_metrics = {
            risk_classes[i]: {
                "precision": float(round(per_cls_p[i], 4)),
                "recall": float(round(per_cls_r[i], 4)),
                "f1": float(round(per_cls_f1[i], 4))
            }
            for i in range(len(risk_classes))
        }

        # 3. Entity Preservation Metrics (Section 21)
        total_source_ents = 0
        total_retained_ents = 0
        category_retention = {"gene": [0, 0], "drug": [0, 0], "dosage": [0, 0], "adverse_event": [0, 0]}

        for i in range(total):
            pred_text = predictions[i].lower()
            ents = reference_entities_list[i] if i < len(reference_entities_list) else {}
            for cat, key in [("gene", "ner_genes"), ("drug", "ner_drugs"), ("dosage", "ner_dosages"), ("adverse_event", "ner_adverse_events")]:
                raw_list = ents.get(key, [])
                for e in raw_list:
                    e_clean = str(e).strip().lower()
                    if e_clean and e_clean not in ("none/unknown", "none", "unknown"):
                        total_source_ents += 1
                        category_retention[cat][0] += 1
                        if e_clean in pred_text:
                            total_retained_ents += 1
                            category_retention[cat][1] += 1

        overall_retention = float(round(total_retained_ents / total_source_ents, 4)) if total_source_ents > 0 else 1.0
        cat_ret_rates = {
            cat: float(round(category_retention[cat][1] / category_retention[cat][0], 4))
            if category_retention[cat][0] > 0 else 1.0
            for cat in category_retention
        }

        # 4. Hallucination / Unsupported Information Check (Section 25)
        hallucinated_records = 0
        for i in range(total):
            pred_text = predictions[i].lower()
            src_note = clinical_notes[i].lower()

            # Check for mention of drugs not in source note
            has_unsupported_drug = any(d in pred_text and d not in src_note for d in self.KNOWN_DRUGS)
            if has_unsupported_drug:
                hallucinated_records += 1

        hallucination_rate = float(round(hallucinated_records / total, 4))

        # 5. Negation Polarity Safety (Section 22)
        negated_cases = 0
        negation_flips = 0
        negation_failures = []

        for i in range(total):
            src_note = clinical_notes[i].lower()
            pred_text = predictions[i].lower()
            pid = patient_ids[i] if patient_ids else f"P_{i}"
            nid = note_ids[i] if note_ids else f"DOC_{i}"

            # Check if source note says 'no acute adverse toxicities' or 'no evidence of'
            if "no acute adverse toxicities" in src_note or "no acute toxicities" in src_note:
                negated_cases += 1
                # Check if predicted text asserts increased hazard
                if "increased" in pred_text and ("hazard" in pred_text or "toxicities" in pred_text) and "no acute" not in pred_text:
                    negation_flips += 1
                    negation_failures.append({
                        "patient_id": pid,
                        "document_id": nid,
                        "source_note": src_note[:200],
                        "expected_polarity": "NEGATED",
                        "predicted_text": pred_text[:200],
                        "detected_entity": "acute adverse toxicities",
                        "issue": "NEGATION_FLIP_AFFIRMED_HAZARD"
                    })

        neg_pres_rate = float(round((negated_cases - negation_flips) / negated_cases, 4)) if negated_cases > 0 else 1.0
        neg_flip_rate = float(round(negation_flips / negated_cases, 4)) if negated_cases > 0 else 0.0

        # 6. Risk-Stratified Performance (Section 26)
        stratified = {}
        for tier in risk_classes:
            tier_indices = [i for i, r in enumerate(expected_risks) if r == tier]
            if not tier_indices:
                continue
            sub_acc = float(round(sum(1 for idx in tier_indices if pred_risks[idx] == tier) / len(tier_indices), 4))
            sub_compliance = float(round(sum(1 for idx in tier_indices if parsed_preds[idx]["is_compliant"]) / len(tier_indices), 4))
            stratified[tier] = {
                "count": len(tier_indices),
                "accuracy": sub_acc,
                "format_compliance": sub_compliance
            }

        return {
            "format_compliance_rate": format_compliance,
            "missing_risk_rate": float(round(missing_risk, 4)),
            "missing_key_finding_rate": float(round(missing_kf, 4)),
            "missing_action_rate": float(round(missing_action, 4)),
            "risk_accuracy": acc,
            "risk_macro_precision": float(round(prec, 4)),
            "risk_macro_recall": float(round(rec, 4)),
            "risk_macro_f1": float(round(f1, 4)),
            "risk_per_class": per_class_metrics,
            "confusion_matrix": conf_mat,
            "entity_retention_rate": overall_retention,
            "entity_retention_by_category": cat_ret_rates,
            "unsupported_entity_rate": hallucination_rate,
            "negation_cases_evaluated": negated_cases,
            "negation_flips": negation_flips,
            "negation_preservation_rate": neg_pres_rate,
            "negation_flip_rate": neg_flip_rate,
            "negation_failures": negation_failures,
            "risk_stratified_evaluation": stratified
        }
