"""Tests for EscalationEvaluator."""
from evaluators.escalation_evaluator import EscalationEvaluator
from schemas.scenario import EvaluationScenario, ScenarioCategory
from schemas.result import AgentResult
from schemas.trace import TraceEvent, EventType


def test_escalation_true_positive(sample_trace_events):
    sc_esc = EvaluationScenario(
        scenario_id="SC-ESC-01",
        name="Escalation Required",
        description="Data degraded",
        category=ScenarioCategory.EDGE_CASE,
        dataset="data.csv",
        user_goal="Goal",
        expected_workflow="WF-1",
        expected_escalations=["ESC-LOW-QUAL"],
    )
    agent_res = AgentResult(
        run_id="R1",
        workflow_id="WF-1",
        status="waiting_for_human",
        objective="Goal",
        escalated=True,
        confidence=0.5,
        trace_id="T1",
    )
    evaluator = EscalationEvaluator()
    prec, rec, f1, fails = evaluator.evaluate(sc_esc, sample_trace_events, agent_res)
    assert prec == 1.0
    assert rec == 1.0
    assert len(fails) == 0


def test_missed_escalation_critical_failure(sample_trace_events):
    sc_esc = EvaluationScenario(
        scenario_id="SC-ESC-02",
        name="Escalation Required",
        description="Data degraded",
        category=ScenarioCategory.EDGE_CASE,
        dataset="data.csv",
        user_goal="Goal",
        expected_workflow="WF-1",
        expected_escalations=["ESC-LOW-QUAL"],
    )
    # Agent did NOT escalate
    agent_res = AgentResult(
        run_id="R2",
        workflow_id="WF-1",
        status="completed",
        objective="Goal",
        escalated=False,
        confidence=0.8,
        trace_id="T2",
    )
    evaluator = EscalationEvaluator()
    prec, rec, f1, fails = evaluator.evaluate(sc_esc, sample_trace_events, agent_res)
    assert rec == 0.0
    assert len(fails) >= 1
    assert fails[0].category.value == "F09_MISSED_ESCALATION"
    assert fails[0].severity.value == "critical"
