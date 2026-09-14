"""End-to-End Evaluation Demo for AADA Agentic AI.

Runs a complete benchmark evaluation on the Revenue Decline scenario,
evaluating Process, Outcome, and Safety against Ground Truth.
"""

import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
EVAL_DIR = CURRENT_DIR.parent
STAGE6_DIR = EVAL_DIR.parent

sys.path.insert(0, str(STAGE6_DIR / "knowledge-engineer"))
sys.path.insert(0, str(STAGE6_DIR / "workflow-engineer"))
sys.path.insert(0, str(STAGE6_DIR / "agent-engineer"))
sys.path.insert(0, str(EVAL_DIR))

# Ensure module aliasing
import types
if "workflow_engineer" not in sys.modules:
    wf_pkg = types.ModuleType("workflow_engineer")
    wf_pkg.__path__ = [str(STAGE6_DIR / "workflow-engineer")]
    sys.modules["workflow_engineer"] = wf_pkg

import schemas
for d in [EVAL_DIR, STAGE6_DIR / "agent-engineer", STAGE6_DIR / "workflow-engineer"]:
    sp = str(d / "schemas")
    if (d / "schemas").exists() and sp not in schemas.__path__:
        schemas.__path__.append(sp)

from schemas.result import AgentResult
from schemas.trace import TraceEvent, EventType
from schemas.scenario import EvaluationScenario
from schemas.ground_truth import GroundTruth
from experiments.runner import ExperimentRunner
from reports.report_generator import ReportGenerator


def build_sample_revenue_run() -> tuple[EvaluationScenario, AgentResult, list[TraceEvent], GroundTruth]:
    """Construct a high-fidelity executed run of revenue decline analysis."""
    scenario = EvaluationScenario(
        scenario_id="SC-REV-001",
        name="Revenue Decline Analysis",
        description="Investigate why sales revenue declined by 18% in Q2/Q3 across enterprise tiers.",
        category="sales",
        dataset="data/sales_q3.csv",
        user_goal="Determine why revenue decreased last quarter and identify the primary driver.",
        expected_workflow="WF-REVENUE-001",
        expected_tasks=["T001", "T002", "T003", "T005", "T006", "T011", "T012", "T013"],
        expected_tools=["load_dataset", "profile_dataset", "validate_data_quality", "eda_analysis", "root_cause_analysis", "generate_recommendation", "generate_report"],
        expected_knowledge=["revenue drop analysis methods"],
        expected_branches=[{"condition_id": "COND-01", "target": "T005"}],
        expected_escalations=[],
    )

    agent_result = AgentResult(
        run_id="RUN-84F2A10C",
        workflow_id="WF-REVENUE-001",
        status="completed",
        objective="Determine why revenue decreased last quarter",
        findings=[
            {"cause": "Price Increase in Enterprise Tier", "score": 0.86, "details": "Decline rate 0.18, magnitude $180,000"}
        ],
        evidence=[
            {"decline_rate": 0.18, "decline_magnitude": 180000.0, "baseline_revenue": 1000000.0}
        ],
        recommendations=[
            {"recommendation": "Rebalance Enterprise pricing tiers to reduce tier-migration churn."},
            {"recommendation": "Introduce volume-based grandfathered discounts for accounts >$50k ARR."}
        ],
        confidence=0.84,
        escalated=False,
        completed_tasks=["T001", "T002", "T003", "T005", "T006", "T011", "T012", "T013"],
        trace_id="TRACE-ROOT-001",
        metrics={"duration_seconds": 0.42, "tool_calls": 8, "llm_calls": 8},
    )

    trace_events = [
        TraceEvent(trace_id="T1", run_id="RUN-84F2A10C", timestamp="2026-09-13T12:00:00Z", task_id="T001", event_type=EventType.TASK_STARTED),
        TraceEvent(trace_id="T2", run_id="RUN-84F2A10C", timestamp="2026-09-13T12:00:01Z", task_id="T001", event_type=EventType.TOOL_CALL, tool_name="load_dataset"),
        TraceEvent(trace_id="T3", run_id="RUN-84F2A10C", timestamp="2026-09-13T12:00:02Z", task_id="T002", event_type=EventType.TOOL_CALL, tool_name="profile_dataset"),
        TraceEvent(trace_id="T4", run_id="RUN-84F2A10C", timestamp="2026-09-13T12:00:03Z", task_id="T003", event_type=EventType.TOOL_CALL, tool_name="validate_data_quality"),
        TraceEvent(trace_id="T5", run_id="RUN-84F2A10C", timestamp="2026-09-13T12:00:04Z", task_id="T003", event_type=EventType.BRANCH_DECISION, decision="continue_analysis", next_task_id="T005"),
        TraceEvent(trace_id="T6", run_id="RUN-84F2A10C", timestamp="2026-09-13T12:00:05Z", task_id="T005", event_type=EventType.TOOL_CALL, tool_name="retrieve_knowledge", tool_input={"query": "revenue drop analysis methods"}),
        TraceEvent(trace_id="T7", run_id="RUN-84F2A10C", timestamp="2026-09-13T12:00:06Z", task_id="T005", event_type=EventType.TOOL_CALL, tool_name="eda_analysis"),
        TraceEvent(trace_id="T8", run_id="RUN-84F2A10C", timestamp="2026-09-13T12:00:07Z", task_id="T006", event_type=EventType.TOOL_CALL, tool_name="root_cause_analysis"),
        TraceEvent(trace_id="T9", run_id="RUN-84F2A10C", timestamp="2026-09-13T12:00:08Z", task_id="T012", event_type=EventType.TOOL_CALL, tool_name="generate_recommendation"),
        TraceEvent(trace_id="T10", run_id="RUN-84F2A10C", timestamp="2026-09-13T12:00:09Z", task_id="T013", event_type=EventType.TOOL_CALL, tool_name="generate_report"),
    ]

    gt = GroundTruth(
        ground_truth_id="GT-REV-001",
        scenario_id="SC-REV-001",
        expected_facts=[
            {"fact_id": "F-01", "description": "Decline rate", "metric_name": "decline_rate", "expected_value": 0.18, "tolerance_pct": 0.05, "is_critical": True},
            {"fact_id": "F-02", "description": "Decline magnitude", "metric_name": "decline_magnitude", "expected_value": 180000.0, "tolerance_pct": 0.05, "is_critical": True}
        ],
        acceptable_causes=[
            {"cause": "Price Increase in Enterprise Tier", "min_score": 0.70, "max_rank": 1}
        ],
        forbidden_claims=["Revenue increased during Q2 and Q3"],
        expected_recommendation_themes=["pricing", "tier", "discount"],
    )

    return scenario, agent_result, trace_events, gt


def main():
    runner = ExperimentRunner()
    scenario, agent_result, trace_events, gt = build_sample_revenue_run()

    result = runner.evaluate_run(
        scenario=scenario,
        agent_result=agent_result,
        trace_events=trace_events,
        ground_truth=gt,
    )

    p = result.process_metrics
    o = result.outcome_metrics
    s = result.safety_metrics

    print("\n============================================================")
    print("AADA AGENT EVALUATION")
    print("============================================================\n")
    print("Scenario:")
    print(f"{scenario.name}\n")
    print("Expected Workflow:")
    print(f"{scenario.expected_workflow}\n")
    print("Actual Run:")
    print(f"{agent_result.run_id}\n")
    print("------------------------------------------------------------")
    print("PROCESS")
    print("------------------------------------------------------------\n")
    print(f"Workflow Adherence       {p.workflow_adherence:.0%}")
    print(f"Tool Selection           {p.tool_selection_accuracy:.0%}")
    print(f"Knowledge Retrieval      {p.knowledge_retrieval_accuracy:.0%}")
    print(f"Branch Accuracy          {p.branch_accuracy:.0%}")
    print(f"Escalation Recall        {p.escalation_recall:.0%}")
    print(f"Evidence Grounding       {p.evidence_grounding:.0%}\n")
    print("------------------------------------------------------------")
    print("OUTCOME")
    print("------------------------------------------------------------\n")
    print(f"Analytical Accuracy      {o.analytical_accuracy:.0%}")
    print(f"Recommendation Quality   {o.recommendation_relevance:.0%}\n")
    print("------------------------------------------------------------")
    print("SAFETY")
    print("------------------------------------------------------------\n")
    print(f"Critical Violations     {s.critical_violations}")
    print(f"High Violations         {s.high_violations}\n")
    print("------------------------------------------------------------\n")
    print("FINAL:\n")
    verdict = "PASS" if result.passed else "FAIL"
    print(verdict)
    print("\n============================================================")

    # Save reports
    overall = runner.run_suite([(scenario, agent_result, trace_events)])
    saved = ReportGenerator.save_reports(overall, output_dir=EVAL_DIR / "reports")
    print(f"\nGenerated JSON report:     {saved['json']}")
    print(f"Generated Markdown report: {saved['markdown']}\n")


if __name__ == "__main__":
    main()
