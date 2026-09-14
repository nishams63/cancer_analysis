"""Conditional branch routing evaluator."""
from typing import List, Tuple, Dict, Any
from schemas.scenario import EvaluationScenario
from schemas.failure import FailureRecord, FailureSeverity, FailureCategory
from schemas.trace import TraceEvent, EventType


class BranchEvaluator:
    """Evaluates runtime dynamic branch decisions against scenario expectations."""

    def evaluate(
        self,
        scenario: EvaluationScenario,
        trace_events: List[TraceEvent],
        run_id: str = "RUN-001",
    ) -> Tuple[float, List[FailureRecord]]:
        failures: List[FailureRecord] = []
        expected_branches = scenario.expected_branches

        if not expected_branches:
            return 1.0, []

        branch_events = [e for e in trace_events if e.event_type == EventType.BRANCH_DECISION]
        matched_branches = 0

        for exp in expected_branches:
            target = exp.get("target")
            found = False
            for be in branch_events:
                if target and be.next_task_id == target:
                    found = True
                    break
                elif not target and be.decision:
                    found = True
                    break

            if found:
                matched_branches += 1
            else:
                failures.append(
                    FailureRecord(
                        failure_id=f"FAIL-BRANCH-{scenario.scenario_id}",
                        scenario_id=scenario.scenario_id,
                        run_id=run_id,
                        category=FailureCategory.F08_WRONG_BRANCH,
                        severity=FailureSeverity.HIGH,
                        expected=f"Branch routing to {exp}",
                        actual="Branch condition was not taken or routed to unexpected target",
                        evidence=f"Recorded branch events: {[b.model_dump() for b in branch_events]}",
                        explanation=f"Agent failed to execute expected conditional branch targeting {target}.",
                        recommended_fix="Verify branch condition threshold against runtime context variables.",
                    )
                )

        branch_accuracy = round(matched_branches / len(expected_branches), 4)
        return branch_accuracy, failures
