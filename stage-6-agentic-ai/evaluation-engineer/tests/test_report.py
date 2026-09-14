"""Tests for report generation."""
from reports.report_generator import ReportGenerator
from experiments.runner import ExperimentRunner


def test_markdown_and_json_report_generation(runner, sample_scenario, sample_agent_result, sample_trace_events):
    overall = runner.run_suite([(sample_scenario, sample_agent_result, sample_trace_events)])
    json_str = ReportGenerator.generate_json(overall)
    md_str = ReportGenerator.generate_markdown(overall)

    assert "evaluation_id" in json_str
    assert "AADA AGENT EVALUATION REPORT" in md_str
    assert "PROCESS METRICS" in md_str
    assert "OUTCOME METRICS" in md_str
    assert "SAFETY & CRITICAL INVARIANTS" in md_str
