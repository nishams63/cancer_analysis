"""
Out-of-Distribution (OOD) Benchmarking and Performance Degradation Module.
Evaluates model across Standard In-Distribution, OOD-Synthetic, and OOD-Real datasets,
measuring degradation in Risk Macro-F1, Entity Retention, Negation Preservation, and Format Compliance.
Per User Guidance: OOD-Synthetic and OOD-Real are explicitly distinguished with provenance.
"""

import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

logger = logging.getLogger("stage6_eval.ood_evaluator")


class OODEvaluator:
    """Evaluates generalization and measures performance degradation under OOD conditions."""

    def __init__(self):
        pass

    def evaluate_cohort(
        self,
        cohort_name: str,
        provenance: str,
        df: pd.DataFrame,
        predictions: List[str]
    ) -> Dict[str, Any]:
        """Evaluates a single cohort on clinical, structural, and safety dimensions."""
        n_samples = len(df)
        if n_samples == 0 or len(predictions) != n_samples:
            raise ValueError(f"Cohort {cohort_name} length mismatch: {n_samples} vs {len(predictions)}")

        expected_risks = [str(r).strip().capitalize() for r in df["expected_risk"]]
        pred_risks = []
        format_compliances = []
        entity_retentions = []
        negation_flips = 0
        negation_cases = 0
        hallucinations = 0

        for i, pred in enumerate(predictions):
            row = df.iloc[i]
            # Format check
            lines = [l.strip() for l in pred.strip().split("\n") if l.strip()]
            has_risk = any(l.startswith("Risk:") for l in lines)
            has_kf = any(l.startswith("Key Finding:") for l in lines)
            has_act = any(l.startswith("Action:") for l in lines)

            fmt_ok = has_risk and has_kf and has_act
            format_compliances.append(1.0 if fmt_ok else 0.0)

            # Risk extraction
            extracted_risk = "Low"
            for l in lines:
                if l.startswith("Risk:"):
                    val = l.replace("Risk:", "").strip().capitalize()
                    if val in ["Low", "Moderate", "High"]:
                        extracted_risk = val
                    break
            pred_risks.append(extracted_risk)

            # Entity retention check
            ref_drugs = [str(d).lower().strip() for d in row.get("ner_drugs", [])]
            pred_lower = pred.lower()
            if ref_drugs:
                ret = sum(1 for d in ref_drugs if d in pred_lower) / len(ref_drugs)
                entity_retentions.append(ret)
            else:
                entity_retentions.append(1.0)

            # Negation safety check
            source_lower = str(row.get("prompt", "")).lower()
            if any(k in source_lower for k in ["no acute adverse", "denies adverse", "without acute toxicity", "freedom from"]):
                negation_cases += 1
                if any(bad in pred_lower for bad in ["developed neutropenia", "increased hazard", "developed severe"]):
                    negation_flips += 1

            # Hallucination check
            if "doxorubicin" in pred_lower and "doxorubicin" not in source_lower:
                hallucinations += 1

        # Compute metrics
        risk_acc = float(accuracy_score(expected_risks, pred_risks))
        labels = ["Low", "Moderate", "High"]
        risk_f1 = float(f1_score(expected_risks, pred_risks, labels=labels, average="macro", zero_division=0))
        fmt_rate = float(np.mean(format_compliances))
        ret_rate = float(np.mean(entity_retentions))
        flip_rate = float(negation_flips / negation_cases) if negation_cases > 0 else 0.0
        hal_rate = float(hallucinations / n_samples)

        return {
            "cohort_name": cohort_name,
            "provenance": provenance,
            "sample_count": n_samples,
            "risk_macro_f1": float(round(risk_f1, 4)),
            "risk_accuracy": float(round(risk_acc, 4)),
            "format_compliance_rate": float(round(fmt_rate, 4)),
            "entity_retention_rate": float(round(ret_rate, 4)),
            "negation_cases": negation_cases,
            "negation_flips": negation_flips,
            "negation_flip_rate": float(round(flip_rate, 4)),
            "hallucination_rate": float(round(hal_rate, 4))
        }

    def compute_degradation(
        self,
        standard_metrics: Dict[str, Any],
        ood_metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculates Delta degradation = Standard - OOD."""
        return {
            "delta_risk_macro_f1": float(round(standard_metrics["risk_macro_f1"] - ood_metrics["risk_macro_f1"], 4)),
            "delta_risk_accuracy": float(round(standard_metrics["risk_accuracy"] - ood_metrics["risk_accuracy"], 4)),
            "delta_format_compliance": float(round(standard_metrics["format_compliance_rate"] - ood_metrics["format_compliance_rate"], 4)),
            "delta_entity_retention": float(round(standard_metrics["entity_retention_rate"] - ood_metrics["entity_retention_rate"], 4)),
            "delta_negation_flips": float(round(ood_metrics["negation_flip_rate"] - standard_metrics["negation_flip_rate"], 4))
        }
