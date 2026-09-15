"""Allowlisted Clinical Analytical Tool Adapters for Oncology CDSS."""
from __future__ import annotations
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime

# Knowledge Engineer retrieval import if available
try:
    from knowledge_engineer.tools.retrieve_knowledge import retrieve_knowledge as ke_retrieve_knowledge
except ImportError:
    ke_retrieve_knowledge = None


# Curated Oncology Guideline Evidence Bank
CLINICAL_KNOWLEDGE_BANK = [
    {
        "knowledge_id": "GUIDE-NSCLC-EGFR-001",
        "cancer_type": "NSCLC",
        "stage": "IV",
        "biomarker": "EGFR_EXON_19",
        "title": "NCCN Guidelines: Metastatic NSCLC with EGFR Sensitizing Mutation",
        "recommendation": "Preferred first-line therapy is Osimertinib 80mg orally once daily (Category 1). FLAURA trial demonstrated superior progression-free and overall survival compared to 1st-generation TKIs (erlotinib/gefitinib) with lower CNS progression.",
        "evidence_level": "Category 1 / Level A",
        "source": "NCCN Non-Small Cell Lung Cancer Guidelines v3.2024",
        "version": "2024.1",
        "contraindications": ["Severe Interstitial Lung Disease (ILD) / Pneumonitis"],
        "alternatives": ["Erlotinib + Ramucirumab", "Afatinib", "Gefitinib"],
    },
    {
        "knowledge_id": "GUIDE-NSCLC-PDL1-002",
        "cancer_type": "NSCLC",
        "stage": "IV",
        "biomarker": "PD-L1_HIGH",
        "title": "NCCN Guidelines: Metastatic NSCLC with PD-L1 TPS >= 50% without EGFR/ALK",
        "recommendation": "Preferred first-line monotherapy is Pembrolizumab 200mg IV every 3 weeks (Category 1, KEYNOTE-024/042). If rapid disease progression or symptomatic high tumor burden, platinum-doublet chemotherapy plus pembrolizumab may be preferred.",
        "evidence_level": "Category 1 / Level A",
        "source": "NCCN Non-Small Cell Lung Cancer Guidelines v3.2024",
        "version": "2024.1",
        "contraindications": ["Active severe autoimmune disease requiring systemic immunosuppression"],
        "alternatives": ["Atezolizumab monotherapy", "Cemiplimab monotherapy", "Carboplatin + Pemetrexed + Pembrolizumab"],
    },
    {
        "knowledge_id": "GUIDE-TOX-CISPLATIN-003",
        "cancer_type": "ALL",
        "stage": "ALL",
        "biomarker": "ALL",
        "title": "Renal Toxicity and Dose Adjustment Protocol: Cisplatin",
        "recommendation": "Cisplatin is strictly contraindicated in patients with significant renal impairment (Creatinine Clearance < 50-60 mL/min or serum creatinine > 1.8-2.0 mg/dL). Carboplatin (AUC-based Calvert dosing) is the recommended substitution.",
        "evidence_level": "Category 1 / Standard of Care",
        "source": "ASCO Clinical Practice Guideline on Chemotherapy in Renal Impairment",
        "version": "2023.2",
        "contraindications": ["Pre-existing renal impairment (CrCl < 50 mL/min)", "Pre-existing severe hearing loss"],
        "alternatives": ["Carboplatin AUC 5-6 substitution"],
    },
    {
        "knowledge_id": "GUIDE-NSCLC-KRAS-004",
        "cancer_type": "NSCLC",
        "stage": "IV",
        "biomarker": "KRAS_G12C",
        "title": "Targeted Therapy for KRAS G12C-Mutated Advanced NSCLC",
        "recommendation": "For patients with KRAS G12C mutation who have received at least one prior systemic therapy, Sotorasib 960mg daily or Adagrasib 600mg BID is recommended based on CodeBreaK 100 / KRYSTAL-1.",
        "evidence_level": "Category 2A / Level B",
        "source": "NCCN Non-Small Cell Lung Cancer Guidelines v3.2024",
        "version": "2024.1",
        "contraindications": ["Severe hepatic impairment (ALT/AST > 5x ULN)"],
        "alternatives": ["Docetaxel + Ramucirumab"],
    },
]


def validate_patient_facts(
    patient_id: str,
    cancer_type: Optional[str] = None,
    stage: Optional[str] = None,
    biomarkers: Optional[Dict[str, Any]] = None,
    laboratory_results: Optional[Dict[str, Any]] = None,
    comorbidities: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Allowlisted tool: Ingest and validate patient clinical facts from trusted EMR sources."""
    missing_critical = []
    if not cancer_type:
        missing_critical.append("cancer_type")
    if not stage:
        missing_critical.append("stage")

    facts_validated = {
        "patient_id": patient_id,
        "cancer_type": cancer_type or "MISSING_INFORMATION",
        "stage": stage or "MISSING_INFORMATION",
        "biomarkers": biomarkers or {},
        "laboratory_results": laboratory_results or {},
        "comorbidities": comorbidities or [],
        "validation_timestamp": datetime.utcnow().isoformat() + "Z",
    }

    is_valid = len(missing_critical) == 0
    return {
        "status": "success" if is_valid else "warning",
        "is_valid": is_valid,
        "missing_critical_facts": missing_critical,
        "provenance": "OBSERVED_FACT",
        "facts": facts_validated,
    }


def retrieve_clinical_guidelines(
    query: str,
    cancer_type: Optional[str] = None,
    stage: Optional[str] = None,
    biomarkers: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    **kwargs,
) -> Dict[str, Any]:
    """Allowlisted tool: Retrieve evidence-grounded clinical guidelines and trial data."""
    # First query Knowledge Engineer if available
    ke_results = []
    if ke_retrieve_knowledge:
        try:
            ke_resp = ke_retrieve_knowledge(query=query, top_k=top_k)
            if ke_resp.get("status") == "success":
                ke_results = ke_resp.get("results", [])
        except Exception:
            ke_results = []

    # Match against curated clinical oncology bank
    matched_guidelines = []
    q_lower = query.lower()
    c_type = (cancer_type or "").upper()
    stg = (stage or "").upper()

    for item in CLINICAL_KNOWLEDGE_BANK:
        score = 0.0
        if c_type and (item["cancer_type"] == c_type or item["cancer_type"] == "ALL"):
            score += 0.4
        if stg and (item["stage"] == stg or item["stage"] == "ALL"):
            score += 0.2
        if biomarkers:
            for b_name, b_val in biomarkers.items():
                if b_name.upper() in item["biomarker"] or str(b_val).upper() in item["biomarker"]:
                    score += 0.4
        if item["cancer_type"].lower() in q_lower or item["title"].lower() in q_lower:
            score += 0.3

        if score > 0.3 or not (cancer_type or stage or biomarkers):
            matched_guidelines.append({
                "knowledge_id": item["knowledge_id"],
                "title": item["title"],
                "source": item["source"],
                "version": item["version"],
                "evidence_level": item["evidence_level"],
                "recommendation": item["recommendation"],
                "contraindications": item["contraindications"],
                "alternatives": item["alternatives"],
                "relevance_score": min(round(score if score > 0 else 0.75, 2), 1.0),
            })

    matched_guidelines.sort(key=lambda x: x["relevance_score"], reverse=True)
    return {
        "status": "success",
        "query": query,
        "retrieved_count": len(matched_guidelines[:top_k]),
        "guidelines": matched_guidelines[:top_k],
        "ke_matches": len(ke_results),
    }


def analyze_clinical_evidence(
    patient_facts: Dict[str, Any],
    retrieved_guidelines: List[Dict[str, Any]],
    **kwargs,
) -> Dict[str, Any]:
    """Allowlisted tool: Synthesize candidate options against patient clinical profile."""
    options = []
    biomarkers = patient_facts.get("biomarkers", {})
    labs = patient_facts.get("laboratory_results", {})
    comorbidities = [c.lower() for c in patient_facts.get("comorbidities", [])]

    for idx, guide in enumerate(retrieved_guidelines):
        opt_id = f"H{idx+1}"
        title = guide.get("title", f"Option {idx+1}")
        rec = guide.get("recommendation", "")
        evidence_level = guide.get("evidence_level", "Standard of Care")
        relevance = guide.get("relevance_score", 0.8)

        # Baseline alignment
        supporting = [f"Recommended by {guide.get('source', 'Clinical Guideline')} ({evidence_level})"]
        contradicting = []

        # Check contraindications
        for ci in guide.get("contraindications", []):
            ci_lower = ci.lower()
            for comorb in comorbidities:
                if comorb in ci_lower or ci_lower in comorb:
                    contradicting.append(f"Contraindicated: Patient comorbidity '{comorb}' conflicts with '{ci}'")
            if "renal" in ci_lower:
                cr = labs.get("creatinine") or labs.get("serum_creatinine")
                crcl = labs.get("crcl") or labs.get("creatinine_clearance")
                if (cr and float(cr) >= 1.8) or (crcl and float(crcl) < 50.0):
                    contradicting.append(f"Contraindicated: Patient renal impairment (Cr={cr}) conflicts with '{ci}'")

        patient_comp = 1.0 if not contradicting else 0.2
        confidence = round(max(0.1, relevance * patient_comp - (len(contradicting) * 0.3)), 2)

        options.append({
            "hypothesis_id": opt_id,
            "option_title": title,
            "recommendation_summary": rec,
            "supporting_evidence": supporting,
            "contradicting_evidence": contradicting,
            "evidence_level": evidence_level,
            "relevance": relevance,
            "patient_compatibility": patient_comp,
            "confidence": confidence,
        })

    return {
        "status": "success",
        "candidate_options": options,
        "synthesized_at": datetime.utcnow().isoformat() + "Z",
    }


def check_contraindications(
    patient_facts: Dict[str, Any],
    candidate_options: List[Dict[str, Any]],
    **kwargs,
) -> Dict[str, Any]:
    """Allowlisted tool: Detect clinical contradictions, organ toxicity, and safety flags."""
    contradictions = []
    labs = patient_facts.get("laboratory_results", {})
    comorbidities = [c.lower() for c in patient_facts.get("comorbidities", [])]

    # Rule 1: Renal Impairment vs Platinum/Cisplatin
    cr = labs.get("creatinine") or labs.get("serum_creatinine")
    if cr and float(cr) >= 1.8:
        for opt in candidate_options:
            opt_title = opt.get("option_title", "").lower()
            if "cisplatin" in opt_title or "platinum" in opt_title:
                contradictions.append({
                    "conflict_id": f"CONF-RENAL-{len(contradictions)+1}",
                    "conflict_type": "FACT_VS_OPTION",
                    "severity": "CRITICAL",
                    "statement_a": f"Patient has serum creatinine {cr} mg/dL indicating severe renal impairment.",
                    "statement_b": f"Proposed option '{opt.get('option_title')}' requires normal renal clearance.",
                    "requires_escalation": True,
                })

    # Rule 2: Interstitial Lung Disease vs EGFR-TKI
    for com in comorbidities:
        if "interstitial lung" in com or "ild" in com or "pulmonary fibrosis" in com:
            for opt in candidate_options:
                opt_title = opt.get("option_title", "").lower()
                if "osimertinib" in opt_title or "egfr" in opt_title:
                    contradictions.append({
                        "conflict_id": f"CONF-ILD-{len(contradictions)+1}",
                        "conflict_type": "FACT_VS_OPTION",
                        "severity": "HIGH",
                        "statement_a": f"Patient has documented comorbidity: {com}.",
                        "statement_b": f"Option '{opt.get('option_title')}' carries risk of fatal drug-induced pneumonitis/ILD.",
                        "requires_escalation": True,
                    })

    # Rule 3: Conflicting Patient Facts (e.g. Stage I vs Distant Metastases)
    stage = str(patient_facts.get("stage", "")).upper()
    pathology = " ".join(patient_facts.get("pathology_findings", [])).lower()
    imaging = " ".join(patient_facts.get("imaging_findings", [])).lower()
    if ("stage i" in stage or stage == "I" or stage == "IA" or stage == "IB") and ("distant metastases" in imaging or "bone metastases" in imaging or "liver metastases" in imaging):
        contradictions.append({
            "conflict_id": f"CONF-STAGE-{len(contradictions)+1}",
            "conflict_type": "FACT_VS_FACT",
            "severity": "CRITICAL",
            "statement_a": f"Recorded clinical stage is {stage}.",
            "statement_b": f"Imaging report demonstrates distant metastatic disease ({imaging[:60]}).",
            "requires_escalation": True,
        })

    return {
        "status": "success",
        "contradictions_found": len(contradictions),
        "contradictions": contradictions,
        "requires_human_escalation": any(c["severity"] in ["CRITICAL", "HIGH"] for c in contradictions),
    }


def calculate_clinical_uncertainty(
    evidence_strength: float = 0.8,
    consistency: float = 1.0,
    missing_critical_facts: Optional[List[str]] = None,
    contradiction_count: int = 0,
    **kwargs,
) -> Dict[str, Any]:
    """Allowlisted tool: Compute multi-factor clinical uncertainty level."""
    missing = missing_critical_facts or []
    base_confidence = evidence_strength * consistency

    # Penalties
    missing_penalty = len(missing) * 0.35
    contradiction_penalty = contradiction_count * 0.30
    calibrated_confidence = max(0.05, min(1.0, base_confidence - missing_penalty - contradiction_penalty))

    if len(missing) > 0 and contradiction_count > 0:
        level = "CRITICAL"
    elif len(missing) > 0 or contradiction_count > 0 or calibrated_confidence < 0.60:
        level = "HIGH"
    elif calibrated_confidence < 0.78:
        level = "MODERATE"
    else:
        level = "LOW"

    human_review = level in ["HIGH", "CRITICAL"]
    return {
        "status": "success",
        "uncertainty_level": level,
        "calibrated_confidence": round(calibrated_confidence, 2),
        "missing_facts_count": len(missing),
        "contradictions_count": contradiction_count,
        "human_review_required": human_review,
        "rationale": (
            f"Uncertainty graded as {level} based on confidence={calibrated_confidence:.2f}, "
            f"missing_critical={missing}, contradictions={contradiction_count}."
        ),
    }


def format_clinician_summary(
    clinical_question: str,
    patient_facts: Dict[str, Any],
    candidate_options: List[Dict[str, Any]],
    uncertainty_level: str,
    contradictions: List[str],
    human_review_required: bool,
    human_review_reason: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Allowlisted tool: Format structured non-autonomous Clinical Decision Support Summary."""
    disclaimer = (
        "CLINICAL DECISION SUPPORT ONLY: This system does NOT autonomously diagnose or prescribe treatment. "
        "The treating oncologist retains sole clinical responsibility for patient care."
    )
    return {
        "status": "success",
        "clinical_question": clinical_question,
        "patient_facts": patient_facts,
        "candidate_options": candidate_options,
        "uncertainty_level": uncertainty_level,
        "contradictions": contradictions,
        "human_review_required": human_review_required,
        "human_review_reason": human_review_reason,
        "disclaimer": disclaimer,
    }
