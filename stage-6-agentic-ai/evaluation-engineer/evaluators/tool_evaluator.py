"""Tool selection precision, recall, and safety evaluator."""
from typing import List, Tuple, Set, Optional, Dict
from schemas.scenario import EvaluationScenario
from schemas.ground_truth import GroundTruth
from schemas.failure import FailureRecord, FailureSeverity, FailureCategory
from schemas.trace import TraceEvent, EventType


class ToolEvaluator:
    """Evaluates tool selection fidelity, allowlisting, and unexpected invocations."""

    def evaluate(
        self,
        scenario: EvaluationScenario,
        trace_events: List[TraceEvent],
        ground_truth: Optional[GroundTruth] = None,
        run_id: str = "RUN-001",
    ) -> Tuple[float, float, float, List[FailureRecord]]:
        failures: List[FailureRecord] = []
        expected_tools = set(scenario.expected_tools)
        executed_tools = [e.tool_name for e in trace_events if e.event_type == EventType.TOOL_CALL and e.tool_name]
        executed_set = set(executed_tools)

        alt_map: Dict[str, List[str]] = ground_truth.acceptable_alternative_tools if ground_truth else {}

        # 1. Precision: proportion of executed tools that are valid
        valid_count = 0
        for tool in executed_set:
            if tool in expected_tools:
                valid_count += 1
            else:
                # Check if it's an accepted alternative
                is_alt = False
                for exp_tool, alts in alt_map.items():
                    if exp_tool in expected_tools and tool in alts:
                        is_alt = True
                        break
                if is_alt:
                    valid_count += 1
                else:
                    # Unexpected tool
                    failures.append(
                        FailureRecord(
                            failure_id=f"FAIL-TOOL-UNEXPECTED-{tool}",
                            scenario_id=scenario.scenario_id,
                            run_id=run_id,
                            category=FailureCategory.F05_WRONG_TOOL,
                            severity=FailureSeverity.MEDIUM,
                            expected=f"Tools in {list(expected_tools)}",
                            actual=f"Unexpected tool '{tool}' executed",
                            evidence=f"Executed tools: {executed_tools}",
                            explanation=f"Agent invoked '{tool}' which is not part of expected or acceptable tools.",
                            recommended_fix="Refine LLM tool selection prompts and task allowlists.",
                        )
                    )

        tool_precision = round(valid_count / len(executed_set), 4) if executed_set else 1.0

        # 2. Recall: proportion of expected tools that were called
        covered_count = 0
        for exp in expected_tools:
            if exp in executed_set:
                covered_count += 1
            else:
                alts = alt_map.get(exp, [])
                if any(alt in executed_set for alt in alts):
                    covered_count += 1

        tool_recall = round(covered_count / len(expected_tools), 4) if expected_tools else 1.0
        tool_accuracy = round((tool_precision + tool_recall) / 2.0, 4)

        return tool_accuracy, tool_precision, tool_recall, failures
