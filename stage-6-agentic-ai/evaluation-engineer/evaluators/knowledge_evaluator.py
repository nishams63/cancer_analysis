"""Knowledge retrieval accuracy and relevance evaluator."""
from typing import List, Tuple, Optional
from schemas.scenario import EvaluationScenario
from schemas.failure import FailureRecord, FailureSeverity, FailureCategory
from schemas.trace import TraceEvent, EventType


class KnowledgeEvaluator:
    """Evaluates whether knowledge was retrieved when required and relevant."""

    def evaluate(
        self,
        scenario: EvaluationScenario,
        trace_events: List[TraceEvent],
        run_id: str = "RUN-001",
    ) -> Tuple[float, float, List[FailureRecord]]:
        failures: List[FailureRecord] = []
        expected_queries = scenario.expected_knowledge

        # Find knowledge retrieval events
        kr_events = [
            e for e in trace_events
            if e.tool_name == "retrieve_knowledge" and e.event_type in (EventType.TOOL_CALL, EventType.OBSERVATION)
        ]

        if not expected_queries:
            return 1.0, 1.0, []

        if not kr_events:
            failures.append(
                FailureRecord(
                    failure_id=f"FAIL-KNOW-MISSING-{scenario.scenario_id}",
                    scenario_id=scenario.scenario_id,
                    run_id=run_id,
                    category=FailureCategory.F06_MISSING_KNOWLEDGE_RETRIEVAL,
                    severity=FailureSeverity.HIGH,
                    expected=f"Knowledge retrieval for {expected_queries}",
                    actual="No retrieve_knowledge calls logged in trace",
                    evidence="Trace events contain 0 retrieve_knowledge invocations.",
                    explanation="Scenario required domain knowledge guidance but agent did not consult knowledge base.",
                    recommended_fix="Include knowledge_retrieval spec in task definition.",
                )
            )
            return 0.0, 0.0, failures

        # Calculate keyword relevance match
        matched_queries = 0
        for exp in expected_queries:
            exp_terms = exp.lower().split()
            found = False
            for event in kr_events:
                inp = str(event.tool_input or {}).lower()
                out = str(event.tool_output_summary or {}).lower()
                if any(term in inp or term in out for term in exp_terms):
                    found = True
                    break
            if found:
                matched_queries += 1

        accuracy = round(matched_queries / len(expected_queries), 4)
        relevance = 1.0 if accuracy >= 0.80 else accuracy

        return accuracy, relevance, failures
