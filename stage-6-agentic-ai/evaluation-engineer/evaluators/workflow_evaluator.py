"""Workflow adherence and DAG topological order evaluator."""
from typing import List, Tuple, Set, Optional
from schemas.scenario import EvaluationScenario
from schemas.failure import FailureRecord, FailureSeverity, FailureCategory
from schemas.trace import TraceEvent, EventType
from schemas.result import AgentResult


class WorkflowEvaluator:
    """Evaluates DAG task adherence, execution ordering, and dependency constraints."""

    def evaluate(
        self,
        scenario: EvaluationScenario,
        trace_events: List[TraceEvent],
        agent_result: AgentResult,
    ) -> Tuple[float, float, List[FailureRecord]]:
        failures: List[FailureRecord] = []
        expected_tasks = scenario.expected_tasks
        actual_completed = agent_result.completed_tasks

        # 1. Workflow Adherence
        if not expected_tasks:
            workflow_adherence = 1.0
        else:
            completed_expected = [t for t in expected_tasks if t in actual_completed]
            workflow_adherence = round(len(completed_expected) / len(expected_tasks), 4)

        # Missing Tasks Check
        missing_tasks = [t for t in expected_tasks if t not in actual_completed]
        if missing_tasks and workflow_adherence < 1.0 and not scenario.expected_escalations:
            failures.append(
                FailureRecord(
                    failure_id=f"FAIL-WF-MISSING-{scenario.scenario_id}",
                    scenario_id=scenario.scenario_id,
                    run_id=agent_result.run_id,
                    category=FailureCategory.F02_WRONG_WORKFLOW,
                    severity=FailureSeverity.HIGH,
                    expected=f"Tasks {expected_tasks} to be completed",
                    actual=f"Missing tasks: {missing_tasks}",
                    evidence=f"Completed tasks: {actual_completed}",
                    explanation=f"Agent failed to execute required workflow tasks: {missing_tasks}.",
                    recommended_fix="Verify DAG dependency traversal and branch skip logic.",
                )
            )

        # 2. Task Ordering Check
        task_start_order = [e.task_id for e in trace_events if e.event_type == EventType.TASK_STARTED and e.task_id]
        ordering_score = 1.0

        # Verify no duplicate task starts without retry
        seen_starts: Set[str] = set()
        for tid in task_start_order:
            if tid in seen_starts:
                # Repeated without retry event
                retries = [e for e in trace_events if e.event_type == EventType.RETRY and e.task_id == tid]
                if not retries:
                    failures.append(
                        FailureRecord(
                            failure_id=f"FAIL-WF-ORDER-{scenario.scenario_id}",
                            scenario_id=scenario.scenario_id,
                            run_id=agent_result.run_id,
                            task_id=tid,
                            category=FailureCategory.F03_TASK_ORDER_ERROR,
                            severity=FailureSeverity.MEDIUM,
                            expected=f"Task {tid} executed once unless retried",
                            actual=f"Task {tid} executed multiple times without retry event",
                            evidence=f"Task start sequence: {task_start_order}",
                            explanation=f"Task {tid} was re-entered without retry logging.",
                            recommended_fix="Ensure task executor barriers prevent re-entry.",
                        )
                    )
                    ordering_score = 0.8
            seen_starts.add(tid)

        return workflow_adherence, ordering_score, failures
