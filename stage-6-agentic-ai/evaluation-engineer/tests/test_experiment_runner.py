"""Tests for ExperimentRunner and version comparison."""
from experiments.runner import ExperimentRunner


def test_runner_evaluates_sample_run(runner, sample_scenario, sample_agent_result, sample_trace_events, sample_ground_truth):
    res = runner.evaluate_run(
        scenario=sample_scenario,
        agent_result=sample_agent_result,
        trace_events=sample_trace_events,
        ground_truth=sample_ground_truth,
    )
    assert res.passed is True
    assert res.overall_score >= 0.85
    assert res.safety_metrics.critical_violations == 0


def test_version_comparison(runner, sample_scenario, sample_agent_result, sample_trace_events):
    overall_v1 = runner.run_suite([(sample_scenario, sample_agent_result, sample_trace_events)])
    overall_v2 = runner.run_suite([(sample_scenario, sample_agent_result, sample_trace_events)])

    comp = runner.compare_versions(overall_v1, overall_v2)
    assert "success_rate_delta" in comp
    assert comp["success_rate_delta"] == 0.0
