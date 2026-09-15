"""Clinical Decision Support Presenter."""
from __future__ import annotations
from typing import Dict, Any, List
from schemas.deliberation import (
    ClinicalDecisionSupportSummary,
    DeliberationState,
)


class ClinicalPresenter:
    """Formats evidence-grounded Clinical Decision Support Summary."""

    def present(self, state: DeliberationState) -> ClinicalDecisionSupportSummary:
        """Render authoritative structured summary with mandatory disclaimer."""
        patient = state.patient_context

        facts = [
            {"label": "Patient ID", "value": patient.patient_id},
            {"label": "Age / Sex", "value": f"{patient.age or 'Unknown'} / {patient.sex or 'Unknown'}"},
            {"label": "Primary Cancer", "value": patient.cancer_type or "MISSING_INFORMATION"},
            {"label": "Clinical Stage", "value": patient.stage or "MISSING_INFORMATION"},
            {"label": "Biomarkers", "value": patient.biomarkers or "None detected"},
            {"label": "Laboratory Results", "value": patient.laboratory_results or "None"},
            {"label": "Comorbidities", "value": patient.comorbidities or "None reported"},
        ]

        evidence = [
            {
                "knowledge_id": ev.knowledge_id,
                "source": ev.source,
                "title": ev.title,
                "recommendation": ev.content,
                "relevance": ev.relevance_score,
            }
            for ev in state.retrieved_evidence
        ]

        options = [
            {
                "id": comp.hypothesis_id,
                "option_title": comp.option_title,
                "confidence": f"{comp.confidence:.0%}",
                "supporting_evidence": comp.supporting_evidence,
                "contradicting_evidence": comp.contradicting_evidence,
                "rank": comp.rank,
            }
            for comp in state.comparisons
        ]

        contradiction_summaries = [
            f"[{c.severity}] {c.statement_a} vs {c.statement_b}" for c in state.contradictions
        ]

        clinician_support = (
            f"Based on validated clinical guidelines for {patient.cancer_type or 'Oncology'} "
            f"(Stage {patient.stage or 'Unknown'}), candidate options were synthesized and compared. "
            f"Top ranked candidate is '{options[0]['option_title'] if options else 'None'}' "
            f"with confidence {options[0]['confidence'] if options else 'N/A'}. "
            f"Uncertainty is graded as {state.uncertainty.overall_level.value}. "
            "Please review all evidence and contraindication warnings before selecting patient management."
        )

        return ClinicalDecisionSupportSummary(
            clinical_question=state.clinical_question,
            patient_facts=facts,
            relevant_evidence=evidence,
            candidate_options=options,
            uncertainty_level=state.uncertainty.overall_level,
            uncertainty_summary=state.uncertainty.rationale,
            contradictions_detected=contradiction_summaries,
            human_review_required=state.human_review_required,
            human_review_reason=state.human_review_reason,
            clinician_decision_support=clinician_support,
        )

    def render_markdown(self, summary: ClinicalDecisionSupportSummary) -> str:
        """Generate human-readable clinical decision support summary text."""
        lines = [
            "============================================================",
            "CLINICAL DECISION SUPPORT SUMMARY",
            "============================================================",
            "",
            "Clinical Question:",
            summary.clinical_question,
            "",
            "------------------------------------------------------------",
            "PATIENT FACTS",
            "------------------------------------------------------------",
        ]
        for f in summary.patient_facts:
            lines.append(f"{f['label']}: {f['value']}")

        lines.extend([
            "",
            "------------------------------------------------------------",
            "RELEVANT EVIDENCE",
            "------------------------------------------------------------",
        ])
        for ev in summary.relevant_evidence:
            lines.append(f"- [{ev['knowledge_id']}] {ev['source']}: {ev['recommendation']}")

        lines.extend([
            "",
            "------------------------------------------------------------",
            "POSSIBLE INTERPRETATIONS / OPTIONS",
            "------------------------------------------------------------",
        ])
        for opt in summary.candidate_options:
            lines.append(f"Option {opt['id']}: {opt['option_title']}")
            lines.append(f"Confidence: {opt['confidence']}")
            lines.append(f"Supporting Evidence: {', '.join(opt['supporting_evidence'])}")
            if opt['contradicting_evidence']:
                lines.append(f"Contradicting Evidence: {', '.join(opt['contradicting_evidence'])}")
            lines.append("")

        lines.extend([
            "------------------------------------------------------------",
            "UNCERTAINTY",
            "------------------------------------------------------------",
            f"Level: {summary.uncertainty_level.value}",
            f"Summary: {summary.uncertainty_summary}",
            "",
            "------------------------------------------------------------",
            "SAFETY / CONTRADICTIONS",
            "------------------------------------------------------------",
        ])
        if summary.contradictions_detected:
            for c in summary.contradictions_detected:
                lines.append(f"- {c}")
        else:
            lines.append("No active contraindications or clinical contradictions detected.")

        lines.extend([
            "",
            "------------------------------------------------------------",
            "HUMAN REVIEW",
            "------------------------------------------------------------",
            f"Required: {'YES' if summary.human_review_required else 'NO'}",
        ])
        if summary.human_review_reason:
            lines.append(f"Reason: {summary.human_review_reason}")

        lines.extend([
            "",
            "------------------------------------------------------------",
            "CLINICIAN DECISION SUPPORT",
            "------------------------------------------------------------",
            summary.clinician_decision_support,
            "",
            "IMPORTANT:",
            summary.disclaimer,
            "============================================================",
        ])
        return "\n".join(lines)
