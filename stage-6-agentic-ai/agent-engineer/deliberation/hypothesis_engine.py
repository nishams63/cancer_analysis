"""Hypothesis Generation and Evidence Comparison Engine."""
from __future__ import annotations
from typing import Dict, Any, List, Optional
from schemas.deliberation import (
    ClinicalHypothesis,
    HypothesisComparison,
    EvidenceItem,
    PatientContext,
    FactProvenance,
)


class HypothesisEngine:
    """Generates competing clinical options and executes multi-factor comparisons."""

    def generate_hypotheses(
        self,
        patient_context: PatientContext,
        evidence: List[EvidenceItem],
    ) -> List[ClinicalHypothesis]:
        """Generate multiple plausible clinical hypotheses/treatment options."""
        hypotheses = []
        for idx, ev in enumerate(evidence):
            h_id = f"H{idx+1}"
            hypotheses.append(
                ClinicalHypothesis(
                    hypothesis_id=h_id,
                    option_title=ev.title,
                    description=ev.content,
                    clinical_rationale=f"Supported by evidence {ev.knowledge_id} ({ev.source}).",
                    provenance=FactProvenance.HYPOTHESIS,
                )
            )

        if len(hypotheses) == 1:
            hypotheses.append(
                ClinicalHypothesis(
                    hypothesis_id="H2",
                    option_title="Standard Chemotherapy / Clinical Trial Alternative",
                    description="Alternative systemic regimen or enrollment in an active clinical trial.",
                    clinical_rationale="Alternative option when targeted therapy is contraindicated or unavailable.",
                    provenance=FactProvenance.HYPOTHESIS,
                )
            )
        elif len(hypotheses) == 0:
            hypotheses.append(
                ClinicalHypothesis(
                    hypothesis_id="H1",
                    option_title="Standard of Care Chemotherapy / Palliative Support",
                    description="Conventional systemic chemotherapy or supportive care depending on performance status.",
                    clinical_rationale="Baseline standard of care in the absence of actionable targeted mutations.",
                    provenance=FactProvenance.HYPOTHESIS,
                )
            )
        return hypotheses

    def compare_hypotheses(
        self,
        hypotheses: List[ClinicalHypothesis],
        patient_context: PatientContext,
        evidence: List[EvidenceItem],
    ) -> List[HypothesisComparison]:
        """Score and rank hypotheses across supporting and contradicting evidence."""
        comparisons = []
        labs = patient_context.laboratory_results or {}
        comorbidities = [c.lower() for c in patient_context.comorbidities]

        for hyp in hypotheses:
            supporting = []
            contradicting = []

            for ev in evidence:
                if ev.title.lower() in hyp.option_title.lower() or hyp.option_title.lower() in ev.title.lower():
                    supporting.append(f"{ev.source}: {ev.content[:80]}...")

            opt_text = (hyp.option_title + " " + hyp.description).lower()
            if "cisplatin" in opt_text:
                cr = labs.get("creatinine") or labs.get("serum_creatinine")
                if cr and float(cr) >= 1.8:
                    contradicting.append(f"Renal Impairment: Creatinine {cr} mg/dL contraindicates Cisplatin.")

            if "osimertinib" in opt_text or "egfr" in opt_text:
                for com in comorbidities:
                    if "interstitial lung" in com or "ild" in com:
                        contradicting.append(f"Pulmonary Risk: Comorbidity '{com}' carries fatal pneumonitis risk.")

            evidence_strength = 0.9 if supporting else 0.5
            relevance = 0.85 if supporting else 0.4
            consistency = 0.9 if not contradicting else 0.2
            uncertainty_score = 0.15 if supporting and not contradicting else 0.5
            contradiction_penalty = 0.4 if contradicting else 0.0

            composite = (
                0.35 * evidence_strength
                + 0.25 * relevance
                + 0.25 * consistency
                - 0.15 * uncertainty_score
                - contradiction_penalty
            )
            confidence = max(0.05, min(0.98, round(composite, 2)))

            comparisons.append(
                HypothesisComparison(
                    hypothesis_id=hyp.hypothesis_id,
                    option_title=hyp.option_title,
                    supporting_evidence=supporting or [hyp.clinical_rationale],
                    contradicting_evidence=contradicting,
                    evidence_strength=evidence_strength,
                    relevance=relevance,
                    patient_compatibility=1.0 if not contradicting else 0.2,
                    consistency=consistency,
                    uncertainty_score=uncertainty_score,
                    contradiction_penalty=contradiction_penalty,
                    confidence=confidence,
                )
            )

        comparisons.sort(key=lambda x: x.confidence, reverse=True)
        for idx, comp in enumerate(comparisons):
            comp.rank = idx + 1

        return comparisons
