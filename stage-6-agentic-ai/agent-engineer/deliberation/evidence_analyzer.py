"""Evidence Analyzer for Deliberative Oncology AI."""
from __future__ import annotations
import uuid
from typing import Dict, Any, List, Optional
from schemas.deliberation import EvidenceItem, PatientContext
from tools.clinical_adapters import retrieve_clinical_guidelines


class EvidenceAnalyzer:
    """Retrieves and analyzes clinical knowledge while maintaining strict provenance."""

    def identify_evidence_needs(
        self,
        question: str,
        patient_context: PatientContext,
    ) -> List[str]:
        """Determine specific clinical evidence requirements."""
        needs = [f"Guideline evidence for {patient_context.cancer_type or 'Oncology'}"]
        if patient_context.stage:
            needs.append(f"Stage {patient_context.stage} management protocols")
        if patient_context.biomarkers:
            for b_name, b_val in patient_context.biomarkers.items():
                needs.append(f"Targeted therapy protocols for {b_name} ({b_val})")
        if patient_context.comorbidities:
            for com in patient_context.comorbidities:
                needs.append(f"Toxicity/Contraindication profiles for comorbidity: {com}")
        return needs

    def retrieve_evidence(
        self,
        question: str,
        patient_context: PatientContext,
    ) -> List[EvidenceItem]:
        """Query Knowledge Engineer and clinical evidence bank."""
        resp = retrieve_clinical_guidelines(
            query=question,
            cancer_type=patient_context.cancer_type,
            stage=patient_context.stage,
            biomarkers=patient_context.biomarkers,
            top_k=5,
        )

        guidelines = resp.get("guidelines", [])
        items = []
        for g in guidelines:
            item = EvidenceItem(
                evidence_id=f"EVID-{uuid.uuid4().hex[:6].upper()}",
                knowledge_id=g.get("knowledge_id", "K-UNKNOWN"),
                source=g.get("source", "Clinical Practice Guideline"),
                version=g.get("version", "1.0.0"),
                title=g.get("title", "Clinical Recommendation"),
                content=g.get("recommendation", ""),
                relevance_score=g.get("relevance_score", 0.8),
                category="clinical_oncology",
            )
            items.append(item)
        return items
