"""Pydantic schemas and data contracts for Deliberative Oncology AI Agent."""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class DeliberationStage(str, Enum):
    """The 12 formal stages of the Deliberative Agent reasoning cycle."""
    UNDERSTAND = "UNDERSTAND"
    PLAN = "PLAN"
    GATHER_EVIDENCE = "GATHER_EVIDENCE"
    ANALYZE = "ANALYZE"
    COMPARE = "COMPARE"
    CHECK_CONTRADICTIONS = "CHECK_CONTRADICTIONS"
    ASSESS_UNCERTAINTY = "ASSESS_UNCERTAINTY"
    DECIDE_NEXT_ACTION = "DECIDE_NEXT_ACTION"
    REQUEST_HUMAN_REVIEW = "REQUEST_HUMAN_REVIEW"
    EXECUTE_ACTION = "EXECUTE_ACTION"
    VERIFY_RESULT = "VERIFY_RESULT"
    PRESENT_CONCLUSION = "PRESENT_CONCLUSION"


class QuestionClassification(str, Enum):
    """Clinical request classification taxonomy."""
    DIAGNOSTIC_REASONING_SUPPORT = "diagnostic_reasoning_support"
    RISK_ASSESSMENT = "risk_assessment"
    TREATMENT_OPTION_REVIEW = "treatment_option_review"
    PROGNOSIS_ANALYSIS = "prognosis_analysis"
    PATHOLOGY_INTERPRETATION = "pathology_interpretation"
    PATIENT_STRATIFICATION = "patient_stratification"
    ADVERSE_EVENT_ANALYSIS = "adverse_event_analysis"
    LITERATURE_SYNTHESIS = "literature_synthesis"
    UNKNOWN = "unknown"


class FactProvenance(str, Enum):
    """Strict evidentiary provenance classification for clinical information."""
    OBSERVED_FACT = "OBSERVED_FACT"              # Directly provided by trusted electronic medical records
    RETRIEVED_KNOWLEDGE = "RETRIEVED_KNOWLEDGE"  # Retrieved from validated knowledge base / clinical guidelines
    DERIVED_INFERENCE = "DERIVED_INFERENCE"      # Calculated deterministically from facts and rules
    HYPOTHESIS = "HYPOTHESIS"                    # Plausible candidate option under deliberation
    UNCERTAINTY = "UNCERTAINTY"                  # Identified knowledge or data gap
    RECOMMENDATION = "RECOMMENDATION"            # Proposed decision support for clinician review
    MISSING_INFORMATION = "MISSING_INFORMATION"  # Explicitly missing required clinical fact


class PatientFact(BaseModel):
    """Single clinical fact with strict provenance and source auditability."""
    name: str = Field(..., description="Fact name e.g. 'cancer_stage', 'egfr_mutation'")
    value: Any = Field(..., description="Fact value e.g. 'IV', 'exon 19 deletion'")
    provenance: FactProvenance = Field(default=FactProvenance.OBSERVED_FACT)
    source: str = Field(default="EMR_STRUCTURED_RECORD", description="Origin of fact")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    verified: bool = Field(default=True)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class PatientContext(BaseModel):
    """Structured patient record from trusted clinical sources."""
    patient_id: str = Field(..., description="Anonymized patient identifier")
    age: Optional[int] = Field(None, ge=0, le=120)
    sex: Optional[str] = Field(None, description="'M', 'F', or 'Other'")
    cancer_type: Optional[str] = Field(None, description="Primary cancer diagnosis e.g. 'NSCLC'")
    stage: Optional[str] = Field(None, description="TNM or Roman stage e.g. 'IV', 'IIIA'")
    biomarkers: Dict[str, Any] = Field(default_factory=dict, description="e.g. {'EGFR': 'positive', 'PD-L1': '80%'}")
    pathology_findings: List[str] = Field(default_factory=list, description="Histology and grade findings")
    imaging_findings: List[str] = Field(default_factory=list, description="CT, PET, MRI summary findings")
    laboratory_results: Dict[str, Any] = Field(default_factory=dict, description="e.g. {'creatinine': 1.1, 'alt': 35}")
    treatment_history: List[Dict[str, Any]] = Field(default_factory=list, description="Prior systemic therapies or surgeries")
    comorbidities: List[str] = Field(default_factory=list, description="Active comorbidities e.g. 'ILD', 'CKD'")
    medications: List[str] = Field(default_factory=list, description="Current outpatient medications")
    raw_facts: List[PatientFact] = Field(default_factory=list, description="Auditable list of all facts")

    def get_fact(self, name: str) -> Optional[Any]:
        """Retrieve fact value or None if missing."""
        if hasattr(self, name):
            val = getattr(self, name)
            if val is not None:
                return val
        if name in self.biomarkers:
            return self.biomarkers[name]
        if name in self.laboratory_results:
            return self.laboratory_results[name]
        for f in self.raw_facts:
            if f.name.lower() == name.lower():
                return f.value
        return None

    def has_fact(self, name: str) -> bool:
        return self.get_fact(name) is not None


class DeliberationPlanStep(BaseModel):
    """Single step in an explicit analytical deliberation plan."""
    step_number: int = Field(..., ge=0)
    objective: str = Field(..., description="Actionable clinical reasoning objective")
    required_evidence: List[str] = Field(default_factory=list)
    safety_checks: List[str] = Field(default_factory=list)
    tool_candidate: Optional[str] = Field(None, description="Allowlisted tool name if needed")
    completed: bool = Field(default=False)
    result_summary: Optional[str] = Field(None)


class DeliberationPlan(BaseModel):
    """Explicit, structured clinical analytical plan."""
    plan_id: str = Field(..., description="Unique plan ID e.g. PLAN-001")
    goal: str = Field(..., description="Clinical question being addressed")
    classification: QuestionClassification = Field(default=QuestionClassification.UNKNOWN)
    steps: List[DeliberationPlanStep] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    is_valid: bool = Field(default=False)
    validation_notes: List[str] = Field(default_factory=list)


class PlanValidationResult(BaseModel):
    """Validation report before executing a deliberation plan."""
    is_valid: bool = Field(..., description="Whether plan satisfies safety and relevance criteria")
    relevance_confirmed: bool = Field(default=True)
    facts_available: bool = Field(default=True)
    missing_required_facts: List[str] = Field(default_factory=list)
    evidence_sources_available: bool = Field(default=True)
    safety_checks_included: bool = Field(default=True)
    human_review_conditions_defined: bool = Field(default=True)
    unnecessary_actions: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    """Unit of clinical evidence retrieved from validated knowledge sources."""
    evidence_id: str = Field(..., description="Unique evidence ID")
    knowledge_id: str = Field(..., description="Source knowledge unit ID in Knowledge Engineer")
    source: str = Field(..., description="Clinical guideline, trial, or publication reference")
    version: str = Field(default="1.0.0")
    title: str = Field(..., description="Summary title of evidence")
    content: str = Field(..., description="Clinical evidence content")
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0)
    category: str = Field(default="clinical_oncology")
    retrieval_timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class ClinicalHypothesis(BaseModel):
    """Plausible clinical interpretation or candidate treatment option."""
    hypothesis_id: str = Field(..., description="e.g. 'H1', 'H2'")
    option_title: str = Field(..., description="Short title e.g. 'Osimertinib 80mg daily'")
    description: str = Field(..., description="Detailed description of hypothesis/option")
    clinical_rationale: str = Field(..., description="Why this hypothesis is plausible")
    biomarker_alignment: Optional[str] = Field(None)
    provenance: FactProvenance = Field(default=FactProvenance.HYPOTHESIS)


class HypothesisComparison(BaseModel):
    """Structured evidence comparison for a candidate hypothesis."""
    hypothesis_id: str = Field(...)
    option_title: str = Field(...)
    supporting_evidence: List[str] = Field(default_factory=list)
    contradicting_evidence: List[str] = Field(default_factory=list)
    evidence_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    patient_compatibility: float = Field(default=1.0, ge=0.0, le=1.0)
    consistency: float = Field(default=1.0, ge=0.0, le=1.0)
    uncertainty_score: float = Field(default=0.2, ge=0.0, le=1.0)
    contradiction_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rank: int = Field(default=1)


class ContradictionType(str, Enum):
    """Clinical conflict taxonomy."""
    FACT_VS_FACT = "FACT_VS_FACT"              # Incompatible clinical findings (e.g. stage I vs distant metastases)
    FACT_VS_OPTION = "FACT_VS_OPTION"          # Contraindication (e.g. renal failure vs cisplatin)
    EVIDENCE_VS_EVIDENCE = "EVIDENCE_VS_EVIDENCE" # Discordant clinical trial results or guidelines
    POLICY_VIOLATION = "POLICY_VIOLATION"      # Violates safety or institutional guideline


class ContradictionRecord(BaseModel):
    """Auditable conflict record identified during deliberation."""
    conflict_id: str = Field(...)
    conflict_type: ContradictionType = Field(...)
    severity: str = Field(default="HIGH", description="'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'")
    statement_a: str = Field(...)
    statement_b: str = Field(...)
    detected_at_stage: DeliberationStage = Field(...)
    resolved: bool = Field(default=False)
    resolution_notes: Optional[str] = Field(None)
    requires_escalation: bool = Field(default=True)


class UncertaintyLevel(str, Enum):
    """Clinical uncertainty grading."""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class UncertaintyAssessment(BaseModel):
    """First-class uncertainty quantification."""
    overall_level: UncertaintyLevel = Field(default=UncertaintyLevel.LOW)
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    epistemic_factors: List[str] = Field(default_factory=list, description="Missing clinical data, unmeasured biomarkers")
    aleatoric_factors: List[str] = Field(default_factory=list, description="Biological heterogeneity, trial population variance")
    missing_critical_facts: List[str] = Field(default_factory=list)
    conflicting_signals: List[str] = Field(default_factory=list)
    human_review_recommended: bool = Field(default=False)
    rationale: str = Field(default="Sufficient concordant clinical evidence.")


class SafetyGateResult(BaseModel):
    """Deterministic validation result for one of the 7 clinical safety gates."""
    gate_number: int = Field(..., ge=1, le=7)
    gate_name: str = Field(..., description="e.g. 'Gate 1: Patient facts validated'")
    passed: bool = Field(...)
    checked_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    blocking: bool = Field(default=True)
    details: str = Field(default="")
    failure_code: Optional[str] = Field(None)


class VerificationResult(BaseModel):
    """Pre-output verification checklist report."""
    claims_supported: bool = Field(default=True)
    facts_accurate: bool = Field(default=True)
    numerical_values_correct: bool = Field(default=True)
    hypotheses_clearly_labeled: bool = Field(default=True)
    uncertainties_stated: bool = Field(default=True)
    contradictions_resolved: bool = Field(default=True)
    human_review_performed: bool = Field(default=True)
    within_clinical_scope: bool = Field(default=True)
    no_autonomous_prescription: bool = Field(default=True)
    all_passed: bool = Field(default=True)
    verification_notes: List[str] = Field(default_factory=list)


class ClinicalDecisionSupportSummary(BaseModel):
    """Structured, non-autonomous clinical decision support output for physician review."""
    clinical_question: str = Field(...)
    patient_facts: List[Dict[str, Any]] = Field(default_factory=list)
    relevant_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    candidate_options: List[Dict[str, Any]] = Field(default_factory=list)
    uncertainty_level: UncertaintyLevel = Field(...)
    uncertainty_summary: str = Field(...)
    contradictions_detected: List[str] = Field(default_factory=list)
    human_review_required: bool = Field(...)
    human_review_reason: Optional[str] = Field(None)
    clinician_decision_support: str = Field(...)
    disclaimer: str = Field(
        default="CLINICAL DECISION SUPPORT ONLY: This system does NOT autonomously diagnose or prescribe treatment. "
                "The treating oncologist retains sole clinical responsibility for patient care."
    )


class DeliberationEvent(BaseModel):
    """Auditable structured event summary (strictly non-private reasoning)."""
    event_id: str = Field(...)
    deliberation_id: str = Field(...)
    step_number: int = Field(...)
    event_name: str = Field(..., description="e.g. 'PLAN_CREATED', 'EVIDENCE_RETRIEVED'")
    stage: DeliberationStage = Field(...)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    summary: str = Field(..., description="High-level factual event summary")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Structured parameters without internal chain-of-thought")


class DeliberationBudget(BaseModel):
    """Defensive limits preventing runaway reasoning cycles."""
    max_deliberation_steps: int = 20
    max_replans: int = 2
    max_tool_calls: int = 10
    max_execution_time_seconds: float = 60.0


class HumanDecision(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REQUEST_MORE_EVIDENCE = "REQUEST_MORE_EVIDENCE"
    OVERRIDE = "OVERRIDE"


class DeliberationState(BaseModel):
    """Authoritative, serializable execution state of a Deliberative Agent run."""
    model_config = {"arbitrary_types_allowed": True}
    deliberation_id: str = Field(..., description="Unique deliberation identifier")
    run_id: str = Field(..., description="High-level orchestration run ID")
    clinical_question: str = Field(..., description="Incoming clinical consultation query")
    question_classification: QuestionClassification = Field(default=QuestionClassification.UNKNOWN)
    patient_context: Any = Field(..., description="Validated patient facts")
    plan: Optional[DeliberationPlan] = Field(None)
    plan_validation: Optional[PlanValidationResult] = Field(None)
    evidence_requirements: List[str] = Field(default_factory=list)
    retrieved_evidence: List[EvidenceItem] = Field(default_factory=list)
    hypotheses: List[ClinicalHypothesis] = Field(default_factory=list)
    comparisons: List[HypothesisComparison] = Field(default_factory=list)
    contradictions: List[ContradictionRecord] = Field(default_factory=list)
    uncertainty: UncertaintyAssessment = Field(default_factory=UncertaintyAssessment)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    safety_flags: List[str] = Field(default_factory=list)
    safety_gates: Dict[str, SafetyGateResult] = Field(default_factory=dict)
    human_review_required: bool = Field(default=False)
    human_review_reason: Optional[str] = Field(None)
    human_decision: Optional[HumanDecision] = Field(None)
    human_override_notes: Optional[str] = Field(None)
    current_stage: DeliberationStage = Field(default=DeliberationStage.UNDERSTAND)
    replan_count: int = Field(default=0)
    tool_call_count: int = Field(default=0)
    verification_result: Optional[VerificationResult] = Field(None)
    final_decision_support: Optional[ClinicalDecisionSupportSummary] = Field(None)
    events: List[DeliberationEvent] = Field(default_factory=list)
    budget: DeliberationBudget = Field(default_factory=DeliberationBudget)
    started_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    completed_at: Optional[str] = Field(None)
    is_halted_for_human: bool = Field(default=False)
    is_failed: bool = Field(default=False)
    failure_reason: Optional[str] = Field(None)

    def record_event(self, event_name: str, summary: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Record an auditable structured event without exposing private chain-of-thought."""
        step_num = len(self.events) + 1
        event = DeliberationEvent(
            event_id=f"EVT-{step_num:03d}",
            deliberation_id=self.deliberation_id,
            step_number=step_num,
            event_name=event_name,
            stage=self.current_stage,
            summary=summary,
            payload=payload or {},
        )
        self.events.append(event)
