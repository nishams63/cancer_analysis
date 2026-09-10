"""
Quality Flag Evaluation and Readiness Decision Engine for Stage 4 EDA.
Applies configurable quality thresholds to measured metrics, assigns PASS/WARNING/CRITICAL
statuses, and synthesizes the final production readiness determination.
"""

from typing import Dict, Any, List, Tuple


class QualityFlagEvaluator:
    """Evaluates audit metrics against configurable engineering and clinical safety thresholds."""

    def __init__(self, thresholds: Dict[str, Any]):
        self.thresholds = thresholds

    def evaluate_all_flags(self, metrics_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates each quality dimension:
        - context_overflow_rate
        - negation_flip_rate
        - entity_retention_rate
        - patient_leakage
        - duplicate_rate
        - risk_class_imbalance
        """
        flags = {}

        # 1. Context Overflow Rate
        overflow_pct = metrics_payload.get("token_overflow_rate", 0.0) / 100.0
        ov_thresh = self.thresholds.get("context_overflow_rate", {"warning": 0.01, "critical": 0.05})
        if overflow_pct >= ov_thresh.get("critical", 0.05):
            status = "CRITICAL"
        elif overflow_pct >= ov_thresh.get("warning", 0.01):
            status = "WARNING"
        else:
            status = "PASS"
        flags["context_overflow"] = {
            "status": status,
            "measured_value": float(round(overflow_pct, 6)),
            "thresholds": ov_thresh,
            "message": f"Context overflow rate is {overflow_pct*100:.2f}%."
        }

        # 2. Negation Flip Rate
        neg_flip_rate = metrics_payload.get("negation_flip_rate", 0.0)
        neg_thresh = self.thresholds.get("negation_flip_rate", {"warning": 0.01, "critical": 0.05})
        if neg_flip_rate >= neg_thresh.get("critical", 0.05):
            status = "CRITICAL"
        elif neg_flip_rate >= neg_thresh.get("warning", 0.01):
            status = "WARNING"
        else:
            status = "PASS"
        flags["negation_flips"] = {
            "status": status,
            "measured_value": float(round(neg_flip_rate, 6)),
            "thresholds": neg_thresh,
            "message": f"Clinical negation flip rate is {neg_flip_rate*100:.2f}%."
        }

        # 3. Entity Retention Rate
        ret_rate = metrics_payload.get("entity_retention_rate", 1.0)
        ret_thresh = self.thresholds.get("entity_retention_rate", {"warning": 0.95, "critical": 0.90})
        if ret_rate < ret_thresh.get("critical", 0.90):
            status = "CRITICAL"
        elif ret_rate < ret_thresh.get("warning", 0.95):
            status = "WARNING"
        else:
            status = "PASS"
        flags["entity_retention"] = {
            "status": status,
            "measured_value": float(round(ret_rate, 6)),
            "thresholds": ret_thresh,
            "message": f"Target entity retention rate is {ret_rate*100:.2f}%."
        }

        # 4. Patient Leakage
        leakage = metrics_payload.get("patient_leakage", 0)
        if leakage > 0:
            status = "CRITICAL"
        else:
            status = "PASS"
        flags["patient_leakage"] = {
            "status": status,
            "measured_value": int(leakage),
            "thresholds": {"critical": 0},
            "message": f"Patient leakage across splits is {leakage} patients."
        }

        # 5. Duplicate Rate
        dup_rate = metrics_payload.get("exact_duplicate_rate", 0.0)
        dup_thresh = self.thresholds.get("duplicate_rate", {"warning": 0.01, "critical": 0.03})
        if dup_rate >= dup_thresh.get("critical", 0.03):
            status = "CRITICAL"
        elif dup_rate >= dup_thresh.get("warning", 0.01):
            status = "WARNING"
        else:
            status = "PASS"
        flags["duplicate_rate"] = {
            "status": status,
            "measured_value": float(round(dup_rate, 6)),
            "thresholds": dup_thresh,
            "message": f"Exact duplicate rate is {dup_rate*100:.2f}%."
        }

        # 6. Risk Class Imbalance
        minority_pct = metrics_payload.get("minority_risk_class_percentage", 0.0) / 100.0
        imb_thresh = self.thresholds.get("risk_class_imbalance", {"warning": 0.20})
        if minority_pct < imb_thresh.get("warning", 0.20):
            status = "WARNING"
        else:
            status = "PASS"
        flags["risk_class_imbalance"] = {
            "status": status,
            "measured_value": float(round(minority_pct, 4)),
            "thresholds": imb_thresh,
            "message": f"Minority risk class represents {minority_pct*100:.2f}% of the dataset."
        }

        # Summary counts
        critical_count = sum(1 for f in flags.values() if f["status"] == "CRITICAL")
        warning_count = sum(1 for f in flags.values() if f["status"] == "WARNING")
        pass_count = sum(1 for f in flags.values() if f["status"] == "PASS")

        # Determine Final Readiness
        if critical_count > 0:
            final_readiness = "NOT READY"
        elif warning_count > 0:
            final_readiness = "READY WITH WARNINGS"
        else:
            final_readiness = "READY"

        decision_reasons = []
        for name, data in flags.items():
            if data["status"] in ("CRITICAL", "WARNING"):
                decision_reasons.append(f"[{data['status']}] {name}: {data['message']}")
        if not decision_reasons:
            decision_reasons.append("All structural, linguistic, entity, and split criteria pass without warning.")

        return {
            "final_status": final_readiness,
            "critical_count": critical_count,
            "warning_count": warning_count,
            "pass_count": pass_count,
            "flags": flags,
            "decision_reasons": decision_reasons
        }
