"""Unit tests for deterministic natural language GoalParser."""
from planner.goal_parser import GoalParser


def test_parse_revenue_decline_goal():
    parser = GoalParser()
    goal = "Analyze this sales dataset and determine why revenue decreased during the last quarter."
    parsed = parser.parse(goal)

    assert parsed.intent_type == "revenue_decline"
    assert parsed.domain == "sales"
    assert parsed.dataset == "sales dataset"
    assert parsed.time_scope == "last quarter"
    assert "revenue trend analysis" in parsed.analysis_requirements
    assert "product analysis" in parsed.analysis_requirements
    assert "root cause analysis and evidence ranking" in parsed.analysis_requirements


def test_parse_customer_churn_goal():
    parser = GoalParser()
    goal = "Analyze customer churn patterns and retention cohort decay in subscriber records."
    parsed = parser.parse(goal)

    assert parsed.intent_type == "customer_analysis"
    assert parsed.domain == "customer"
    assert "churn" in parsed.key_metrics[0] or "churn" in parsed.analysis_requirements[3]


def test_parse_anomaly_goal():
    parser = GoalParser()
    goal = "Detect and investigate unusual transaction spikes and statistical anomalies."
    parsed = parser.parse(goal)

    assert parsed.intent_type == "anomaly_investigation"
    assert parsed.domain == "operations"


def test_parse_predictive_goal():
    parser = GoalParser()
    goal = "Train and evaluate a predictive model to forecast customer conversion."
    parsed = parser.parse(goal)

    assert parsed.intent_type == "predictive_analysis"
    assert parsed.domain == "predictive"


def test_parser_determinism():
    parser = GoalParser()
    goal = "Analyze this sales dataset and determine why revenue decreased during the last quarter."
    first = parser.parse(goal)
    for _ in range(10):
        repeat = parser.parse(goal)
        assert repeat.model_dump() == first.model_dump()
