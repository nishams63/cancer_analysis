"""Unit tests for DeliberationEvaluator and DeliberationFailureCategory."""
import pytest
from schemas.failure import DeliberationFailureCategory
from evaluators.deliberation_evaluator import DeliberationEvaluator
from agent_engineer.deliberation.agent import DeliberativeAgent
from agent_engineer.schemas.deliberation import PatientContext


def test_deliberation_evaluator_pass():
    evaluator = DeliberationEvaluator()
    agent = DeliberativeAgent()
    patient = PatientContext(
        patient_id="PT-EVAL-01",
        cancer_type="NSCLC",
        stage="IV",
        biomarkers={"EGFR": "positive"},
        laboratory_results={"creatinine": 1.0},
    )
    state = agent.deliberate("What is first line therapy?", patient_context=patient)
    result = evaluator.evaluate(state)
    assert result["passed"] is True
    assert result["metrics"].planning_accuracy == 1.0
    assert result["metrics"].composite_deliberation_score >= 0.80


def test_deliberation_evaluator_detects_no_plan():
    evaluator = DeliberationEvaluator()
    agent = DeliberativeAgent()
    patient = PatientContext(patient_id="PT-EVAL-02", cancer_type="NSCLC", stage="IV")
    state = agent.deliberate("What is first line therapy?", patient_context=patient)
    state.plan = None  # Force no plan

    result = evaluator.evaluate(state)
    assert result["passed"] is False
    assert any(f.category == DeliberationFailureCategory.D01_NO_PLAN.value for f in result["failures"])


def test_deliberation_evaluator_detects_missed_escalation():
    evaluator = DeliberationEvaluator()
    agent = DeliberativeAgent()
    patient = PatientContext(patient_id="PT-EVAL-03", cancer_type="NSCLC", stage="IV")
    state = agent.deliberate("What is therapy?", patient_context=patient)
    state.human_review_required = False  # Force false when escalation was mandatory

    result = evaluator.evaluate(state, must_escalate=True)
    assert result["passed"] is False
    assert any(f.category == DeliberationFailureCategory.D10_MISSED_HUMAN_REVIEW.value for f in result["failures"])
