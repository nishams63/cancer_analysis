"""
Adversarial Clinical Testing and Negation Robustness Module.
Evaluates model resilience under semantic paraphrases, drug spelling variations,
abbreviations, and dense negation stress, strictly enforcing Negation Flip Rate thresholds.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger("stage6_eval.adversarial_evaluator")


class AdversarialEvaluator:
    """Evaluates clinical SLM against adversarial perturbations and semantic mutations."""

    SAFETY_NEGATION_THRESHOLD = 0.01 # Max 1.0% allowed negation flip rate

    def __init__(self, threshold: float = SAFETY_NEGATION_THRESHOLD):
        self.threshold = threshold

    def evaluate_adversarial_suite(
        self,
        df: pd.DataFrame,
        predictions: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluates adversarial perturbation suite with granular category breakdown.
        """
        n_total = len(df)
        if n_total != len(predictions):
            raise ValueError(f"Length mismatch: {n_total} records vs {len(predictions)} predictions")

        categories = df["sub_category"].unique() if "sub_category" in df.columns else ["all"]
        cat_metrics = {}

        total_neg_cases = 0
        total_neg_flips = 0
        correct_risks = 0
        preserved_entities = []

        for cat in categories:
            subset_mask = (df["sub_category"] == cat) if "sub_category" in df.columns else np.ones(n_total, dtype=bool)
            sub_indices = np.where(subset_mask)[0]
            sub_n = len(sub_indices)

            sub_flips = 0
            sub_cases = 0
            sub_correct_risk = 0
            sub_retention = []

            for idx in sub_indices:
                row = df.iloc[idx]
                pred = predictions[idx]
                pred_lower = pred.lower()
                source_lower = str(row.get("prompt", "")).lower()
                exp_risk = str(row.get("expected_risk", "Low")).strip().capitalize()

                # Risk check
                pred_risk = "Low"
                for line in pred.split("\n"):
                    if line.strip().startswith("Risk:"):
                        val = line.replace("Risk:", "").strip().capitalize()
                        if val in ["Low", "Moderate", "High"]:
                            pred_risk = val
                        break
                if pred_risk == exp_risk:
                    sub_correct_risk += 1
                    correct_risks += 1

                # Negation check
                is_neg_case = any(p in source_lower for p in [
                    "no acute adverse", "denies adverse", "without acute toxicity",
                    "free of treatment", "zero adverse", "not demonstrate any signs"
                ])
                if is_neg_case:
                    sub_cases += 1
                    total_neg_cases += 1
                    # Check for polarity flip
                    if any(f in pred_lower for f in ["developed neutropenia", "increased hazard", "developed severe", "grade 3", "grade 4"]):
                        sub_flips += 1
                        total_neg_flips += 1

                # Entity check
                ref_drugs = [str(d).lower().strip() for d in row.get("ner_drugs", [])]
                if ref_drugs:
                    ret = sum(1 for d in ref_drugs if d in pred_lower) / len(ref_drugs)
                    sub_retention.append(ret)
                    preserved_entities.append(ret)
                else:
                    sub_retention.append(1.0)
                    preserved_entities.append(1.0)

            cat_metrics[str(cat)] = {
                "sample_count": sub_n,
                "risk_accuracy": float(round(sub_correct_risk / sub_n, 4)) if sub_n > 0 else 0.0,
                "entity_retention_rate": float(round(np.mean(sub_retention), 4)) if sub_retention else 1.0,
                "negation_cases": sub_cases,
                "negation_flips": sub_flips,
                "negation_flip_rate": float(round(sub_flips / sub_cases, 4)) if sub_cases > 0 else 0.0
            }

        overall_flip_rate = float(total_neg_flips / total_neg_cases) if total_neg_cases > 0 else 0.0
        passed_safety = overall_flip_rate <= self.threshold

        return {
            "total_adversarial_samples": n_total,
            "overall_risk_accuracy": float(round(correct_risks / n_total, 4)),
            "overall_entity_retention": float(round(np.mean(preserved_entities), 4)) if preserved_entities else 1.0,
            "total_negation_cases": total_neg_cases,
            "total_negation_flips": total_neg_flips,
            "overall_negation_flip_rate": float(round(overall_flip_rate, 4)),
            "safety_threshold": self.threshold,
            "safety_gate_passed": passed_safety,
            "category_breakdown": cat_metrics
        }
