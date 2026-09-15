"""Uncertainty Assessor for Deliberative Oncology AI."""
from __future__ import annotations
from typing import Dict, Any, List, Optional
from schemas.deliberation import (
    UncertaintyAssessment,
    UncertaintyLevel,
    PatientContext,
    HypothesisComparison,
    ContradictionRecord,
)
from tools.clinical_adapters import calculate_clinical_uncertainty


class UncertaintyAssessor:
    """Quantifies epistemic and aleatoric uncertainty into a first-class metric."""

    def assess(
        self,
        patient_context: PatientContext,
        comparisons: List[HypothesisComparison],
        contradictions: List[ContradictionRecord],
    ) -> UncertaintyAssessment:
        """Calculate calibrated uncertainty score and categorical level."""
        missing_critical = []
        if not patient_context.cancer_type:
            missing_critical.append("cancer_type")
        if not patient_context.stage:
            missing_critical.append("stage")

        best_confidence = comparisons[0].confidence if comparisons else 0.5
        contradiction_count = len(contradictions)

        resp = calculate_clinical_uncertainty(
            evidence_strength=best_confidence,
            consistency=1.0 if contradiction_count == 0 else 0.5,
            missing_critical_facts=missing_critical,
            contradiction_count=contradiction_count,
        )

        level_str = resp.get("uncertainty_level", "MODERATE")
        level = UncertaintyLevel(level_str)
        calibrated_conf = resp.get("calibrated_confidence", 0.5)

        epistemic = []
        if missing_critical:
            epistemic.append(f"Missing clinical variables: {', '.join(missing_critical)}")
        if contradiction_count > 0:
            epistemic.append(f"Identified {contradiction_count} active clinical contradictions")

        aleatoric = [
            "Heterogeneous clinical trial response rates in patient subpopulations",
            "Variable biological pharmacokinetics and drug tolerance",
        ]

        human_review = resp.get("human_review_required", False)
        if level in [UncertaintyLevel.HIGH, UncertaintyLevel.CRITICAL] or contradiction_count > 0 or missing_critical:
            human_review = True

        return UncertaintyAssessment(
            overall_level=level,
            confidence_score=calibrated_conf,
            epistemic_factors=epistemic,
            aleatoric_factors=aleatoric,
            missing_critical_facts=missing_critical,
            conflicting_signals=[f"{c.conflict_id}: {c.statement_a} vs {c.statement_b}" for c in contradictions],
            human_review_recommended=human_review,
            rationale=resp.get("rationale", ""),
        )
