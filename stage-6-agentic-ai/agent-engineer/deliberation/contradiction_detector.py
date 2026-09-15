"""Clinical Contradiction and Conflict Detector."""
from __future__ import annotations
from typing import Dict, Any, List, Optional
from schemas.deliberation import (
    ContradictionRecord,
    ContradictionType,
    DeliberationStage,
    PatientContext,
    ClinicalHypothesis,
)
from tools.clinical_adapters import check_contraindications


class ContradictionDetector:
    """Identifies fact-vs-fact, fact-vs-option, and policy contradictions."""

    def check_contradictions(
        self,
        patient_context: PatientContext,
        hypotheses: List[ClinicalHypothesis],
        current_stage: DeliberationStage = DeliberationStage.CHECK_CONTRADICTIONS,
    ) -> List[ContradictionRecord]:
        """Detect clinical conflicts and return structured auditable records."""
        candidate_options = [{"option_title": h.option_title} for h in hypotheses]
        patient_dict = {
            "stage": patient_context.stage,
            "biomarkers": patient_context.biomarkers,
            "laboratory_results": patient_context.laboratory_results,
            "comorbidities": patient_context.comorbidities,
            "pathology_findings": patient_context.pathology_findings,
            "imaging_findings": patient_context.imaging_findings,
        }

        resp = check_contraindications(patient_facts=patient_dict, candidate_options=candidate_options)
        raw_conflicts = resp.get("contradictions", [])

        records = []
        for c in raw_conflicts:
            records.append(
                ContradictionRecord(
                    conflict_id=c["conflict_id"],
                    conflict_type=ContradictionType(c["conflict_type"]),
                    severity=c.get("severity", "HIGH"),
                    statement_a=c["statement_a"],
                    statement_b=c["statement_b"],
                    detected_at_stage=current_stage,
                    resolved=False,
                    requires_escalation=c.get("requires_escalation", True),
                )
            )
        return records
