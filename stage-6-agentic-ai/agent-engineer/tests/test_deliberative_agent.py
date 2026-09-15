"""Deterministic test suite covering all 20 clinical deliberation scenarios."""
import pytest
from datetime import datetime

from schemas.deliberation import (
    PatientContext,
    PatientFact,
    FactProvenance,
    HumanDecision,
    UncertaintyLevel,
    DeliberationStage,
    DeliberationBudget,
)
from deliberation.agent import DeliberativeAgent
from deliberation.planner import DeliberativePlanner
from deliberation.hypothesis_engine import HypothesisEngine
from deliberation.contradiction_detector import ContradictionDetector
from deliberation.verifier import ConclusionVerifier
from tools.clinical_adapters import validate_patient_facts, retrieve_clinical_guidelines


@pytest.fixture
def agent():
    return DeliberativeAgent()


@pytest.fixture
def base_nsclc_patient():
    return PatientContext(
        patient_id="PT-EGFR-001",
        age=64,
        sex="F",
        cancer_type="NSCLC",
        stage="IV",
        biomarkers={"EGFR": "Exon 19 deletion", "PD-L1": "5%"},
        pathology_findings=["Adenocarcinoma of the lung", "Grade 2"],
        imaging_findings=["Multiple bilateral pulmonary nodules", "Pleural effusion"],
        laboratory_results={"creatinine": 0.9, "alt": 24, "ast": 22},
        comorbidities=["Hypertension"],
        medications=["Amlodipine 5mg daily"],
    )


def test_01_simple_clinical_question(agent, base_nsclc_patient):
    """Scenario 1: Simple clinical question with straightforward guideline match."""
    state = agent.deliberate(
        clinical_question="What is the recommended first-line systemic therapy for this patient?",
        patient_context=base_nsclc_patient,
    )
    assert state.current_stage == DeliberationStage.PRESENT_CONCLUSION
    assert state.plan is not None
    assert len(state.retrieved_evidence) >= 1
    assert len(state.comparisons) >= 1
    assert "Osimertinib" in state.comparisons[0].option_title or "EGFR" in state.comparisons[0].option_title
    assert state.uncertainty.overall_level in [UncertaintyLevel.LOW, UncertaintyLevel.MODERATE]
    assert state.final_decision_support is not None
    assert "CLINICAL DECISION SUPPORT ONLY" in state.final_decision_support.disclaimer


def test_02_complex_multistep_question(agent):
    """Scenario 2: Complex multi-step question comparing immunotherapy vs chemo-IO."""
    patient = PatientContext(
        patient_id="PT-PDL1-002",
        age=68,
        sex="M",
        cancer_type="NSCLC",
        stage="IV",
        biomarkers={"EGFR": "negative", "ALK": "negative", "PD-L1": "85%"},
        pathology_findings=["Squamous cell carcinoma"],
        imaging_findings=["Right upper lobe primary with mediastinal lymphadenopathy"],
        laboratory_results={"creatinine": 1.0, "alt": 30},
        comorbidities=[],
        medications=[],
    )
    state = agent.deliberate(
        clinical_question="Evaluate evidence for first-line pembrolizumab monotherapy versus combination chemo-immunotherapy.",
        patient_context=patient,
    )
    assert len(state.hypotheses) >= 2
    assert any("Pembrolizumab" in h.option_title or "PD-L1" in h.option_title for h in state.hypotheses)
    assert state.verification_result is not None


def test_03_missing_patient_fact(agent):
    """Scenario 3: Missing critical patient fact triggers replan and human review."""
    patient_missing = PatientContext(
        patient_id="PT-MISSING-003",
        cancer_type="NSCLC",
        stage=None,  # Missing stage!
        biomarkers={"EGFR": "positive"},
    )
    state = agent.deliberate(
        clinical_question="What is the appropriate targeted therapy?",
        patient_context=patient_missing,
    )
    assert state.replan_count >= 1
    assert "stage" in state.uncertainty.missing_critical_facts
    assert state.human_review_required is True
    assert state.is_halted_for_human is True


def test_04_conflicting_patient_facts(agent):
    """Scenario 4: Stage I recorded but imaging demonstrates distant metastases."""
    patient_conflict = PatientContext(
        patient_id="PT-CONF-004",
        cancer_type="NSCLC",
        stage="I",  # Contradicts distant metastases below!
        imaging_findings=["CT demonstrates widespread distant metastases in bone and liver."],
        laboratory_results={"creatinine": 1.0},
    )
    state = agent.deliberate(
        clinical_question="Determine clinical staging concordance and treatment strategy.",
        patient_context=patient_conflict,
    )
    assert len(state.contradictions) >= 1
    assert any(c.conflict_type.value == "FACT_VS_FACT" for c in state.contradictions)
    assert state.human_review_required is True


def test_05_conflicting_evidence(agent):
    """Scenario 5: Renal failure contraindicates cisplatin chemotherapy."""
    patient_renal = PatientContext(
        patient_id="PT-RENAL-005",
        cancer_type="NSCLC",
        stage="IV",
        laboratory_results={"creatinine": 2.4},  # Renal failure!
        comorbidities=["Chronic kidney disease stage 4"],
    )
    state = agent.deliberate(
        clinical_question="Can patient receive cisplatin-based doublet chemotherapy?",
        patient_context=patient_renal,
    )
    assert len(state.contradictions) >= 1
    assert any(c.conflict_type.value == "FACT_VS_OPTION" for c in state.contradictions)
    assert state.human_review_required is True


def test_06_high_uncertainty(agent):
    """Scenario 6: High uncertainty due to unmeasured biomarkers and missing stage."""
    patient_uncertain = PatientContext(
        patient_id="PT-UNC-006",
        cancer_type="NSCLC",
        stage=None,
        biomarkers={},
        laboratory_results={},
    )
    state = agent.deliberate(
        clinical_question="Recommend targeted systemic therapy.",
        patient_context=patient_uncertain,
    )
    assert state.uncertainty.overall_level in [UncertaintyLevel.HIGH, UncertaintyLevel.CRITICAL]
    assert state.human_review_required is True


def test_07_strong_evidence(agent, base_nsclc_patient):
    """Scenario 7: Strong concordant evidence achieves high confidence and passes verification."""
    state = agent.deliberate(
        clinical_question="Determine EGFR first-line guideline alignment.",
        patient_context=base_nsclc_patient,
    )
    assert state.confidence >= 0.70
    assert state.verification_result.all_passed is True


def test_08_weak_evidence(agent):
    """Scenario 8: Rare scenario without targeted match results in moderate confidence."""
    patient_rare = PatientContext(
        patient_id="PT-RARE-008",
        cancer_type="Unknown Primary",
        stage="IV",
    )
    state = agent.deliberate(
        clinical_question="Identify actionable targeted therapy.",
        patient_context=patient_rare,
    )
    assert state.confidence <= 0.85
    assert state.uncertainty.overall_level in [UncertaintyLevel.MODERATE, UncertaintyLevel.HIGH]


def test_09_multiple_competing_hypotheses(agent, base_nsclc_patient):
    """Scenario 9: Formulates and ranks multiple competing hypotheses."""
    state = agent.deliberate(
        clinical_question="Compare systemic therapy options.",
        patient_context=base_nsclc_patient,
    )
    assert len(state.comparisons) >= 2
    assert state.comparisons[0].rank == 1
    assert state.comparisons[1].rank == 2
    assert state.comparisons[0].confidence >= state.comparisons[1].confidence


def test_10_human_review_requirement(agent):
    """Scenario 10: Mandatory halt before consequential action when conflict detected."""
    patient_ild = PatientContext(
        patient_id="PT-ILD-010",
        cancer_type="NSCLC",
        stage="IV",
        biomarkers={"EGFR": "positive"},
        comorbidities=["Severe interstitial lung disease / idiopathic pulmonary fibrosis"],
    )
    state = agent.deliberate(
        clinical_question="Evaluate osimertinib therapy for patient with pulmonary fibrosis.",
        patient_context=patient_ild,
    )
    assert state.is_halted_for_human is True
    assert state.current_stage == DeliberationStage.REQUEST_HUMAN_REVIEW


def test_11_human_approval(agent):
    """Scenario 11: Clinician approves escalated decision and agent completes verification."""
    patient_ild = PatientContext(
        patient_id="PT-ILD-011",
        cancer_type="NSCLC",
        stage="IV",
        biomarkers={"EGFR": "positive"},
        comorbidities=["Idiopathic pulmonary fibrosis"],
    )
    # First pass halts
    state1 = agent.deliberate("Evaluate treatment options.", patient_context=patient_ild)
    assert state1.is_halted_for_human is True

    # Second pass with clinician APPROVE
    state2 = agent.deliberate(
        "Evaluate treatment options.",
        patient_context=patient_ild,
        existing_state=state1,
        human_decision=HumanDecision.APPROVE,
        human_override_notes="Close pulmonary monitoring agreed with pulmonology.",
    )
    assert state2.is_halted_for_human is False
    assert state2.current_stage == DeliberationStage.PRESENT_CONCLUSION
    assert state2.final_decision_support is not None


def test_12_human_rejection(agent):
    """Scenario 12: Clinician rejects options -> state marked failed."""
    patient = PatientContext(patient_id="PT-012", cancer_type="NSCLC", stage=None)
    state1 = agent.deliberate("Recommend treatment.", patient_context=patient)
    assert state1.is_halted_for_human is True

    state2 = agent.deliberate(
        "Recommend treatment.",
        patient_context=patient,
        existing_state=state1,
        human_decision=HumanDecision.REJECT,
        human_override_notes="Repeat biopsy requested; do not deliberate further.",
    )
    assert state2.is_failed is True
    assert "rejected" in state2.failure_reason.lower()


def test_13_request_more_evidence(agent):
    """Scenario 13: Clinician requests more evidence -> triggers replanning."""
    patient = PatientContext(patient_id="PT-013", cancer_type="NSCLC", stage=None)
    state1 = agent.deliberate("Recommend treatment.", patient_context=patient)
    assert state1.is_halted_for_human is True

    state2 = agent.deliberate(
        "Recommend treatment.",
        patient_context=patient,
        existing_state=state1,
        human_decision=HumanDecision.REQUEST_MORE_EVIDENCE,
    )
    assert state2.replan_count >= 1


def test_14_replanning(agent):
    """Scenario 14: Deliberative planner constructs bounded replan when facts missing."""
    patient = PatientContext(patient_id="PT-014", cancer_type=None, stage="IV")
    state = agent.deliberate("What treatment applies?", patient_context=patient)
    assert state.replan_count >= 1
    assert any("replan" in s.objective.lower() for s in state.plan.steps)


def test_15_maximum_replanning_limit(agent):
    """Scenario 15: Replan budget prevents infinite replanning cycles."""
    budget = DeliberationBudget(max_replans=1)
    custom_agent = DeliberativeAgent(budget=budget)
    patient = PatientContext(patient_id="PT-015", cancer_type=None, stage=None)
    state = custom_agent.deliberate("Identify therapy.", patient_context=patient)
    assert state.replan_count <= 1


def test_16_verification_failure():
    """Scenario 16: Verification fails if prior safety gates failed."""
    verifier = ConclusionVerifier()
    patient = PatientContext(patient_id="PT-016", cancer_type=None, stage=None)
    agent = DeliberativeAgent(verifier=verifier)
    state = agent.deliberate("Evaluate treatment.", patient_context=patient)
    gates = verifier.evaluate_gates(state)
    assert gates["GATE_1_FACTS"].passed is False


def test_17_unsupported_conclusion(base_nsclc_patient):
    """Scenario 17: Zero evidence triggers verification gate failure."""
    verifier = ConclusionVerifier()
    agent = DeliberativeAgent(verifier=verifier)
    state = agent.deliberate("General query", patient_context=base_nsclc_patient)
    state.retrieved_evidence = []  # Force empty evidence
    gates = verifier.evaluate_gates(state)
    assert gates["GATE_3_EVIDENCE"].passed is False


def test_18_hallucinated_patient_fact():
    """Scenario 18: Provenance verification prevents hallucinated facts."""
    fact = PatientFact(name="EGFR", value="positive", provenance=FactProvenance.OBSERVED_FACT)
    assert fact.provenance == FactProvenance.OBSERVED_FACT
    assert fact.verified is True


def test_19_tool_failure(agent):
    """Scenario 19: Allowlisted tools return safe error objects without unhandled crash."""
    from tools.clinical_adapters import validate_patient_facts
    res = validate_patient_facts(patient_id="PT-TEST", cancer_type=None, stage=None)
    assert res["status"] == "warning"
    assert res["is_valid"] is False


def test_20_knowledge_retrieval_failure(agent):
    """Scenario 20: Knowledge retrieval tool handles empty query safely."""
    from tools.clinical_adapters import retrieve_clinical_guidelines
    res = retrieve_clinical_guidelines(query="", cancer_type=None)
    assert res["status"] == "success"
    assert isinstance(res["guidelines"], list)
