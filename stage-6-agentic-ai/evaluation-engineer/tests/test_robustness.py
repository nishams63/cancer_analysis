"""Tests for counterfactual robustness evaluation."""
from analysis.robustness import RobustnessEvaluator
from schemas.result import AgentResult


def test_counterfactual_responsiveness():
    res_base = AgentResult(
        run_id="R-BASE",
        workflow_id="WF-1",
        status="completed",
        objective="Goal",
        findings=[{"cause": "Volume Drop in Enterprise"}],
        confidence=0.85,
        trace_id="TR-1",
    )
    res_counter = AgentResult(
        run_id="R-COUNTER",
        workflow_id="WF-1",
        status="completed",
        objective="Goal",
        findings=[{"cause": "Price Hike Contraction"}],
        confidence=0.85,
        trace_id="TR-2",
    )
    eval_res = RobustnessEvaluator.evaluate_counterfactual(res_base, res_counter, altered_factor="pricing_vs_volume")
    assert eval_res["is_responsive_to_evidence"] is True
    assert eval_res["robustness_score"] == 1.0
