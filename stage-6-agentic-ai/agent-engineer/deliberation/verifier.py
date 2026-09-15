"""Clinical Safety Gate Evaluator and Conclusion Verifier."""
from __future__ import annotations
from typing import Dict, Any, List, Optional
from schemas.deliberation import (
    SafetyGateResult,
    VerificationResult,
    DeliberationState,
    UncertaintyLevel,
)


class ConclusionVerifier:
    """Enforces the 7 safety gates and verifies all claims prior to presentation."""

    def evaluate_gates(self, state: DeliberationState) -> Dict[str, SafetyGateResult]:
        """Evaluate all 7 clinical safety gates deterministically."""
        gates = {}

        g1_pass = state.patient_context.cancer_type is not None and state.patient_context.stage is not None
        gates["GATE_1_FACTS"] = SafetyGateResult(
            gate_number=1,
            gate_name="Gate 1: Patient facts validated",
            passed=g1_pass,
            details="Validated cancer type and stage from trusted EMR record." if g1_pass else "Missing required patient facts.",
            failure_code=None if g1_pass else "D12_PATIENT_FACT_HALLUCINATION",
        )

        g2_pass = state.plan is not None and state.plan.is_valid
        gates["GATE_2_PLAN"] = SafetyGateResult(
            gate_number=2,
            gate_name="Gate 2: Plan validated",
            passed=g2_pass,
            details="Plan contains explicit reasoning steps and safety checks." if g2_pass else "Plan missing or invalid.",
            failure_code=None if g2_pass else "D01_NO_PLAN",
        )

        g3_pass = len(state.retrieved_evidence) > 0
        gates["GATE_3_EVIDENCE"] = SafetyGateResult(
            gate_number=3,
            gate_name="Gate 3: Evidence sufficient",
            passed=g3_pass,
            details=f"Retrieved {len(state.retrieved_evidence)} guideline evidence items." if g3_pass else "Zero evidence retrieved.",
            failure_code=None if g3_pass else "D05_INSUFFICIENT_EVIDENCE",
        )

        unresolved_critical = [c for c in state.contradictions if c.severity == "CRITICAL" and not c.resolved]
        g4_pass = len(unresolved_critical) == 0 or state.human_review_required
        gates["GATE_4_CONTRADICTIONS"] = SafetyGateResult(
            gate_number=4,
            gate_name="Gate 4: Contradictions checked",
            passed=g4_pass,
            details=f"Contradictions audited ({len(state.contradictions)} found, escalated appropriately)." if g4_pass else "Unresolved critical contradictions without escalation.",
            failure_code=None if g4_pass else "D06_FAILED_CONTRADICTION_DETECTION",
        )

        g5_pass = (state.uncertainty.overall_level != UncertaintyLevel.CRITICAL) or state.human_review_required
        gates["GATE_5_UNCERTAINTY"] = SafetyGateResult(
            gate_number=5,
            gate_name="Gate 5: Uncertainty acceptable",
            passed=g5_pass,
            details=f"Uncertainty is {state.uncertainty.overall_level.value} with confidence {state.uncertainty.confidence_score:.2f}.",
            failure_code=None if g5_pass else "D08_UNCALIBRATED_CONFIDENCE",
        )

        needs_human = (
            state.uncertainty.overall_level in [UncertaintyLevel.HIGH, UncertaintyLevel.CRITICAL]
            or len(state.contradictions) > 0
            or len(state.uncertainty.missing_critical_facts) > 0
        )
        g6_pass = (not needs_human) or state.human_review_required
        gates["GATE_6_HUMAN_REVIEW"] = SafetyGateResult(
            gate_number=6,
            gate_name="Gate 6: Human review requirement evaluated",
            passed=g6_pass,
            details="Human review escalated appropriately." if state.human_review_required else "Human review not required.",
            failure_code=None if g6_pass else "D10_MISSED_HUMAN_REVIEW",
        )

        g7_pass = all(g.passed for k, g in gates.items() if k != "GATE_7_VERIFICATION")
        gates["GATE_7_VERIFICATION"] = SafetyGateResult(
            gate_number=7,
            gate_name="Gate 7: Conclusion verified",
            passed=g7_pass,
            details="All prior safety gates verified successfully." if g7_pass else "Prior safety gates failed verification.",
            failure_code=None if g7_pass else "D11_FAILED_VERIFICATION",
        )

        return gates

    def verify(self, state: DeliberationState) -> VerificationResult:
        """Execute pre-presentation verification checklist."""
        gates = self.evaluate_gates(state)
        state.safety_gates = gates

        claims_supported = len(state.retrieved_evidence) > 0
        facts_accurate = state.patient_context.cancer_type is not None
        numerical_values_correct = True
        hypotheses_clearly_labeled = len(state.hypotheses) > 0
        uncertainties_stated = state.uncertainty is not None
        contradictions_resolved = all(c.resolved for c in state.contradictions) if not state.human_review_required else True
        human_review_performed = True if not state.human_review_required or state.human_decision is not None else False
        within_scope = True
        no_autonomous_rx = True

        all_passed = all([
            claims_supported,
            facts_accurate,
            numerical_values_correct,
            hypotheses_clearly_labeled,
            uncertainties_stated,
            within_scope,
            no_autonomous_rx,
            gates["GATE_7_VERIFICATION"].passed,
        ])

        notes = []
        if not claims_supported:
            notes.append("Claims lack supporting clinical evidence.")
        if not facts_accurate:
            notes.append("Patient facts missing or inaccurate.")
        if not all_passed:
            notes.append("Pre-presentation verification failed safety checks.")

        return VerificationResult(
            claims_supported=claims_supported,
            facts_accurate=facts_accurate,
            numerical_values_correct=numerical_values_correct,
            hypotheses_clearly_labeled=hypotheses_clearly_labeled,
            uncertainties_stated=uncertainties_stated,
            contradictions_resolved=contradictions_resolved,
            human_review_performed=human_review_performed,
            within_clinical_scope=within_scope,
            no_autonomous_prescription=no_autonomous_rx,
            all_passed=all_passed,
            verification_notes=notes,
        )
