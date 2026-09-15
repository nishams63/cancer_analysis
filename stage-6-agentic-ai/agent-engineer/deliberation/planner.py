"""Deliberative Clinical Planner for Oncology Decision Support."""
from __future__ import annotations
import uuid
from typing import Dict, Any, List, Optional
from schemas.deliberation import (
    DeliberationPlan,
    DeliberationPlanStep,
    PlanValidationResult,
    QuestionClassification,
    PatientContext,
)


class DeliberativePlanner:
    """Builds and validates structured analytical plans for clinical questions."""

    def classify_question(self, question: str) -> QuestionClassification:
        """Classify incoming clinical request without assuming treatment is always required."""
        q = question.lower()
        if any(w in q for w in ["treatment", "therapy", "regimen", "drug", "first-line", "second-line", "manage"]):
            return QuestionClassification.TREATMENT_OPTION_REVIEW
        if any(w in q for w in ["diagnos", "biopsy", "pathol", "histol"]):
            return QuestionClassification.PATHOLOGY_INTERPRETATION
        if any(w in q for w in ["prognos", "survival", "recurrence", "hazard"]):
            return QuestionClassification.PROGNOSIS_ANALYSIS
        if any(w in q for w in ["risk", "toxicity", "adverse", "contraindicat"]):
            return QuestionClassification.ADVERSE_EVENT_ANALYSIS
        if any(w in q for w in ["stratify", "subgroup", "cohort"]):
            return QuestionClassification.PATIENT_STRATIFICATION
        if any(w in q for w in ["trial", "evidence", "literature", "study", "nccn"]):
            return QuestionClassification.LITERATURE_SYNTHESIS
        if any(w in q for w in ["stage", "stage iv", "metast"]):
            return QuestionClassification.DIAGNOSTIC_REASONING_SUPPORT
        return QuestionClassification.TREATMENT_OPTION_REVIEW

    def create_plan(
        self,
        clinical_question: str,
        patient_context: PatientContext,
        replan_reason: Optional[str] = None,
    ) -> DeliberationPlan:
        """Construct explicit, ordered clinical deliberation plan."""
        classification = self.classify_question(clinical_question)
        plan_id = f"PLAN-{uuid.uuid4().hex[:6].upper()}"

        steps = [
            DeliberationPlanStep(
                step_number=1,
                objective="Validate patient clinical facts (cancer type, stage, biomarkers, labs)",
                required_evidence=["cancer_type", "stage"],
                safety_checks=["Check for missing critical clinical identifiers"],
                tool_candidate="validate_patient_facts",
            ),
            DeliberationPlanStep(
                step_number=2,
                objective="Verify organ function and active comorbidities",
                required_evidence=["creatinine", "alt", "comorbidities"],
                safety_checks=["Identify organ failure and high-risk comorbidities"],
            ),
            DeliberationPlanStep(
                step_number=3,
                objective="Retrieve applicable clinical practice guidelines and evidence",
                required_evidence=["nccn_guidelines", "fda_approvals"],
                safety_checks=["Check evidence recency and clinical indication match"],
                tool_candidate="retrieve_clinical_guidelines",
            ),
            DeliberationPlanStep(
                step_number=4,
                objective="Formulate plausible candidate clinical options/hypotheses",
                required_evidence=["guideline_recommendations"],
                safety_checks=["Ensure at least 2 distinct options or null hypothesis"],
            ),
            DeliberationPlanStep(
                step_number=5,
                objective="Perform structured evidence comparison across candidate options",
                required_evidence=["clinical_trials", "efficacy_metrics"],
                safety_checks=["Evaluate supporting vs contradicting evidence"],
                tool_candidate="analyze_clinical_evidence",
            ),
            DeliberationPlanStep(
                step_number=6,
                objective="Detect contraindications, toxicities, and clinical contradictions",
                required_evidence=["contraindication_matrix", "patient_labs"],
                safety_checks=["Enforce organ toxicity safety rules"],
                tool_candidate="check_contraindications",
            ),
            DeliberationPlanStep(
                step_number=7,
                objective="Quantify epistemic and aleatoric uncertainty",
                required_evidence=["data_completeness", "evidence_concordance"],
                safety_checks=["Flag high uncertainty for mandatory human escalation"],
                tool_candidate="calculate_clinical_uncertainty",
            ),
            DeliberationPlanStep(
                step_number=8,
                objective="Determine human review and escalation necessity",
                required_evidence=["uncertainty_level", "contradiction_records"],
                safety_checks=["Mandate human review for critical decisions or high uncertainty"],
            ),
            DeliberationPlanStep(
                step_number=9,
                objective="Execute pre-output conclusion verification checklist",
                required_evidence=["supported_claims", "provenance_check"],
                safety_checks=["Verify no autonomous diagnosis or prescription is asserted"],
            ),
            DeliberationPlanStep(
                step_number=10,
                objective="Format evidence-grounded Clinical Decision Support Summary",
                required_evidence=["summary_template"],
                safety_checks=["Include mandatory clinical non-autonomous disclaimer"],
                tool_candidate="format_clinician_summary",
            ),
        ]

        if replan_reason:
            steps.insert(
                0,
                DeliberationPlanStep(
                    step_number=1,
                    objective=f"Execute bounded replanning: {replan_reason}",
                    safety_checks=["Confirm replan budget limit not exceeded"],
                ),
            )
            for idx, s in enumerate(steps):
                s.step_number = idx + 1

        plan = DeliberationPlan(
            plan_id=plan_id,
            goal=clinical_question,
            classification=classification,
            steps=steps,
            is_valid=True,
            validation_notes=["Plan conforms to standard 10-step clinical deliberation cycle."],
        )
        return plan

    def validate_plan(
        self,
        plan: DeliberationPlan,
        patient_context: PatientContext,
    ) -> PlanValidationResult:
        """Validate plan relevance, fact availability, and safety constraints."""
        missing_facts = []
        if not patient_context.cancer_type:
            missing_facts.append("cancer_type")
        if not patient_context.stage:
            missing_facts.append("stage")

        has_safety_checks = any(len(s.safety_checks) > 0 for s in plan.steps)
        has_human_review = any("human review" in s.objective.lower() for s in plan.steps)
        relevance_confirmed = len(plan.steps) >= 5

        errors = []
        if not has_safety_checks:
            errors.append("Plan lacks explicit clinical safety checks.")
        if not has_human_review:
            errors.append("Plan lacks mandatory human review evaluation step.")

        is_valid = len(errors) == 0

        return PlanValidationResult(
            is_valid=is_valid,
            relevance_confirmed=relevance_confirmed,
            facts_available=len(missing_facts) == 0,
            missing_required_facts=missing_facts,
            evidence_sources_available=True,
            safety_checks_included=has_safety_checks,
            human_review_conditions_defined=has_human_review,
            unnecessary_actions=[],
            errors=errors,
        )
