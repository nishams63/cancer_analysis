"""Tests for ResultFormatter evidence classification and presentation."""
from presentation.result_formatter import ResultFormatter
from agent_engineer.schemas.result import AgentResult


def test_evidence_classification_and_formatting():
    agent_res = AgentResult(
        run_id="RUN-FMT-001",
        workflow_id="WF-001",
        status="completed",
        objective="Analyze revenue drop",
        findings=[
            {"cause": "Price Increase in Enterprise Tier", "score": 0.85, "details": "decline_rate 0.18"}
        ],
        evidence=[
            {"revenue_decline_magnitude": 180000.0, "rate": 0.18},
            {"hypothesis": "Competitor launched new tier"}
        ],
        recommendations=[{"recommendation": "Rebalance tier pricing"}],
        confidence=0.88,
        completed_tasks=["T001", "T002"],
        trace_id="TR-1",
    )

    formatted = ResultFormatter.format(agent_res, goal="Analyze revenue drop")

    assert "Price Increase in Enterprise Tier" in formatted["summary"]
    assert len(formatted["findings"]) == 1
    assert len(formatted["evidence"]) == 2

    # Verify classification
    ev1 = formatted["evidence"][0]
    ev2 = formatted["evidence"][1]
    assert ev1["classification"] == "OBSERVED FACT"
    assert ev2["classification"] == "EVIDENCE-SUPPORTED INFERENCE"
