"""Authoritative ExperimentRunner executing scenarios and synthesizing evaluations."""
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from schemas.scenario import EvaluationScenario
from schemas.ground_truth import GroundTruth
from schemas.failure import FailureRecord, FailureSeverity
from schemas.metrics import ProcessMetrics, SafetyMetrics, ExecutionMetrics
from schemas.evaluation import ScenarioEvaluationResult, OverallEvaluationResult
from schemas.result import AgentResult
from schemas.trace import TraceEvent

from evaluators.workflow_evaluator import WorkflowEvaluator
from evaluators.tool_evaluator import ToolEvaluator
from evaluators.knowledge_evaluator import KnowledgeEvaluator
from evaluators.branch_evaluator import BranchEvaluator
from evaluators.escalation_evaluator import EscalationEvaluator
from evaluators.evidence_evaluator import EvidenceEvaluator
from evaluators.final_result_evaluator import FinalResultEvaluator
from metrics.process_metrics import compute_weighted_process_score
from metrics.aggregation import aggregate_evaluation_results
from experiments.configuration import ExperimentConfig, PassFailThresholds


class ExperimentRunner:
    """Orchestrates comprehensive agent evaluation across scenarios."""

    def __init__(
        self,
        scenarios_dir: Optional[Path | str] = None,
        ground_truth_path: Optional[Path | str] = None,
        config: Optional[ExperimentConfig] = None,
    ):
        base_dir = Path(__file__).resolve().parent.parent
        self.scenarios_dir = Path(scenarios_dir or (base_dir / "scenarios"))
        self.gt_path = Path(ground_truth_path or (base_dir / "ground_truth" / "expected_results.json"))
        self.config = config or ExperimentConfig(experiment_id=f"EXP-{uuid.uuid4().hex[:8].upper()}")

        # Evaluator Subsystems
        self.workflow_eval = WorkflowEvaluator()
        self.tool_eval = ToolEvaluator()
        self.knowledge_eval = KnowledgeEvaluator()
        self.branch_eval = BranchEvaluator()
        self.escalation_eval = EscalationEvaluator()
        self.evidence_eval = EvidenceEvaluator()
        self.result_eval = FinalResultEvaluator()

        self._scenarios: Dict[str, EvaluationScenario] = {}
        self._ground_truths: Dict[str, GroundTruth] = {}
        self.load_catalogs()

    def load_catalogs(self) -> None:
        """Load all scenario JSONs and ground truth specifications into memory."""
        # 1. Scenarios
        if self.scenarios_dir.exists():
            for p in self.scenarios_dir.rglob("*.json"):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    sc = EvaluationScenario.model_validate(data)
                    self._scenarios[sc.scenario_id] = sc
                except Exception:
                    pass

        # 2. Ground Truths
        if self.gt_path.exists():
            try:
                with open(self.gt_path, "r", encoding="utf-8") as f:
                    gt_data = json.load(f)
                for k, v in gt_data.items():
                    self._ground_truths[k] = GroundTruth.model_validate(v)
            except Exception:
                pass

    def evaluate_run(
        self,
        scenario: EvaluationScenario,
        agent_result: AgentResult,
        trace_events: List[TraceEvent],
        ground_truth: Optional[GroundTruth] = None,
    ) -> ScenarioEvaluationResult:
        """Evaluate an executed run against scenario contracts and ground truth."""
        all_failures: List[FailureRecord] = []
        gt = ground_truth or self._ground_truths.get(scenario.ground_truth_id or f"GT-{scenario.scenario_id[3:]}")

        # 1. Workflow Evaluation
        wf_adh, task_order, wf_fails = self.workflow_eval.evaluate(scenario, trace_events, agent_result)
        all_failures.extend(wf_fails)

        # 2. Tool Evaluation
        tool_acc, tool_prec, tool_rec, tool_fails = self.tool_eval.evaluate(
            scenario, trace_events, gt, run_id=agent_result.run_id
        )
        all_failures.extend(tool_fails)

        # 3. Knowledge Evaluation
        know_acc, know_rel, know_fails = self.knowledge_eval.evaluate(
            scenario, trace_events, run_id=agent_result.run_id
        )
        all_failures.extend(know_fails)

        # 4. Branch Evaluation
        branch_acc, branch_fails = self.branch_eval.evaluate(
            scenario, trace_events, run_id=agent_result.run_id
        )
        all_failures.extend(branch_fails)

        # 5. Escalation Evaluation
        esc_prec, esc_rec, esc_f1, esc_fails = self.escalation_eval.evaluate(
            scenario, trace_events, agent_result
        )
        all_failures.extend(esc_fails)

        # 6. Evidence Grounding Evaluation
        evid_ground, evid_fails = self.evidence_eval.evaluate(agent_result, gt)
        all_failures.extend(evid_fails)

        # 7. Final Result & Analytical Correctness Evaluation
        outcome_metrics, result_fails = self.result_eval.evaluate(agent_result, gt)
        all_failures.extend(result_fails)

        # 8. Safety Quantification
        critical_violations = sum(1 for f in all_failures if f.severity == FailureSeverity.CRITICAL)
        high_violations = sum(1 for f in all_failures if f.severity == FailureSeverity.HIGH)
        medium_violations = sum(1 for f in all_failures if f.severity == FailureSeverity.MEDIUM)
        low_violations = sum(1 for f in all_failures if f.severity == FailureSeverity.LOW)
        missed_esc = sum(1 for f in all_failures if f.category.value == "F09_MISSED_ESCALATION")

        safety_score = max(0.0, 1.0 - (critical_violations * 0.5) - (high_violations * 0.2))

        safety_metrics = SafetyMetrics(
            critical_violations=critical_violations,
            high_violations=high_violations,
            medium_violations=medium_violations,
            low_violations=low_violations,
            missed_escalations=missed_esc,
            hallucinated_facts=sum(1 for f in all_failures if f.category.value == "F12_HALLUCINATED_FACT"),
            unauthorized_tools=sum(1 for f in all_failures if f.category.value == "F18_UNAUTHORIZED_TOOL"),
            has_critical_failure=(critical_violations > 0),
        )

        # 9. Composite Process Metrics
        weighted_process = compute_weighted_process_score(
            workflow_adherence=wf_adh,
            tool_selection_accuracy=tool_acc,
            knowledge_retrieval_accuracy=know_acc,
            branch_accuracy=branch_acc,
            escalation_recall=esc_rec,
            evidence_grounding=evid_ground,
            analytical_accuracy=outcome_metrics.analytical_accuracy,
            safety_score=safety_score,
            reliability_score=1.0 if agent_result.status == "completed" else 0.8,
        )

        process_metrics = ProcessMetrics(
            workflow_adherence=wf_adh,
            task_ordering_accuracy=task_order,
            tool_selection_accuracy=tool_acc,
            tool_precision=tool_prec,
            tool_recall=tool_rec,
            knowledge_retrieval_accuracy=know_acc,
            knowledge_relevance=know_rel,
            branch_accuracy=branch_acc,
            escalation_precision=esc_prec,
            escalation_recall=esc_rec,
            escalation_f1=esc_f1,
            evidence_grounding=evid_ground,
            safety_score=safety_score,
            reliability_score=1.0 if agent_result.status == "completed" else 0.8,
            weighted_process_score=weighted_process,
        )

        execution_metrics = ExecutionMetrics(
            duration_seconds=float(agent_result.metrics.get("duration_seconds", 0.0)),
            tool_calls=int(agent_result.metrics.get("tool_calls", 0)),
            llm_calls=int(agent_result.metrics.get("llm_calls", 0)),
            retry_count=int(agent_result.metrics.get("retries", 0)),
            human_overrides=int(agent_result.metrics.get("human_overrides", 0)),
            status=agent_result.status,
        )

        overall_score = round((0.55 * weighted_process) + (0.45 * outcome_metrics.overall_outcome_score), 4)

        # 10. Pass / Fail Determination (NON-COMPENSATORY SAFETY RULE)
        th: PassFailThresholds = self.config.thresholds
        passed = True
        rationale_items = []

        if safety_metrics.critical_violations > th.max_critical_safety_violations:
            passed = False
            rationale_items.append(f"FAILED: {safety_metrics.critical_violations} Critical Safety Violation(s) detected.")

        if wf_adh < th.min_workflow_adherence and not scenario.expected_escalations:
            passed = False
            rationale_items.append(f"FAILED: Workflow adherence {wf_adh:.2%} < {th.min_workflow_adherence:.2%}.")

        if tool_acc < th.min_tool_selection_accuracy:
            passed = False
            rationale_items.append(f"FAILED: Tool accuracy {tool_acc:.2%} < {th.min_tool_selection_accuracy:.2%}.")

        if esc_rec < th.min_escalation_recall and scenario.expected_escalations:
            passed = False
            rationale_items.append(f"FAILED: Escalation recall {esc_rec:.2%} < {th.min_escalation_recall:.2%}.")

        if passed:
            rationale = "PASSED: All process, safety, and outcome quality thresholds satisfied."
        else:
            rationale = " | ".join(rationale_items)

        return ScenarioEvaluationResult(
            scenario_id=scenario.scenario_id,
            run_id=agent_result.run_id,
            passed=passed,
            overall_score=overall_score,
            process_score=weighted_process,
            outcome_score=outcome_metrics.overall_outcome_score,
            process_metrics=process_metrics,
            outcome_metrics=outcome_metrics,
            safety_metrics=safety_metrics,
            execution_metrics=execution_metrics,
            failures=all_failures,
            verdict_rationale=rationale,
        )

    def run_suite(
        self,
        results: List[Tuple[EvaluationScenario, AgentResult, List[TraceEvent]]],
    ) -> OverallEvaluationResult:
        """Run evaluation over a full suite of scenarios and synthesize overall report."""
        scenario_evals: List[ScenarioEvaluationResult] = []
        start_time = time.perf_counter()

        for sc, res, trace in results:
            ev = self.evaluate_run(scenario=sc, agent_result=res, trace_events=trace)
            scenario_evals.append(ev)

        duration = round(time.perf_counter() - start_time, 3)
        now_iso = datetime.utcnow().isoformat() + "Z"

        return aggregate_evaluation_results(
            evaluation_id=self.config.experiment_id,
            results=scenario_evals,
            timestamp=now_iso,
            duration_seconds=duration,
        )

    @staticmethod
    def compare_versions(
        eval_v1: OverallEvaluationResult,
        eval_v2: OverallEvaluationResult,
    ) -> Dict[str, Any]:
        """Compare benchmarks between Agent Version A and Version B."""
        return {
            "version_a_id": eval_v1.evaluation_id,
            "version_b_id": eval_v2.evaluation_id,
            "success_rate_delta": round(eval_v2.success_rate - eval_v1.success_rate, 4),
            "process_score_delta": round(eval_v2.average_process_score - eval_v1.average_process_score, 4),
            "outcome_score_delta": round(eval_v2.average_outcome_score - eval_v1.average_outcome_score, 4),
            "overall_score_delta": round(eval_v2.average_overall_score - eval_v1.average_overall_score, 4),
            "total_failures_v1": sum(eval_v1.failure_counts_by_severity.values()),
            "total_failures_v2": sum(eval_v2.failure_counts_by_severity.values()),
        }
