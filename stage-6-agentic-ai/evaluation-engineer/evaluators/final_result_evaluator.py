"""Analytical correctness, numerical tolerance, and recommendation quality evaluator."""
import math
from typing import List, Tuple, Optional
from schemas.result import AgentResult
from schemas.ground_truth import GroundTruth
from schemas.metrics import OutcomeMetrics
from schemas.failure import FailureRecord, FailureSeverity, FailureCategory


class FinalResultEvaluator:
    """Evaluates factual agreement, numerical tolerance bounds, and recommendation actionability."""

    def evaluate(
        self,
        agent_result: AgentResult,
        ground_truth: Optional[GroundTruth] = None,
    ) -> Tuple[OutcomeMetrics, List[FailureRecord]]:
        failures: List[FailureRecord] = []
        if not ground_truth:
            metrics = OutcomeMetrics(
                analytical_accuracy=1.0,
                numerical_accuracy=1.0,
                factual_accuracy=1.0,
                root_cause_accuracy=1.0,
                recommendation_relevance=1.0,
                recommendation_actionability=1.0,
                recommendation_grounding=1.0,
                overall_outcome_score=1.0,
            )
            return metrics, []

        # 1. Numerical & Factual Accuracy
        facts = ground_truth.expected_facts
        verified_facts = 0
        findings_str = str(agent_result.findings) + str(agent_result.evidence)

        for fact in facts:
            metric = fact.metric_name
            exp_val = fact.expected_value
            tol = fact.tolerance_pct

            if exp_val is not None and isinstance(exp_val, (int, float)):
                # Search for numerical presence within tolerance
                found_match = False
                lower_bound = exp_val * (1.0 - tol)
                upper_bound = exp_val * (1.0 + tol)

                # Check evidence records
                for ev in agent_result.evidence:
                    if isinstance(ev, dict) and metric in ev:
                        actual_val = ev[metric]
                        if isinstance(actual_val, (int, float)) and lower_bound <= actual_val <= upper_bound:
                            found_match = True
                            break

                if found_match or (str(exp_val) in findings_str):
                    verified_facts += 1
                else:
                    failures.append(
                        FailureRecord(
                            failure_id=f"FAIL-FACT-{fact.fact_id}",
                            scenario_id=ground_truth.scenario_id,
                            run_id=agent_result.run_id,
                            category=FailureCategory.F13_NUMERICAL_ERROR,
                            severity=FailureSeverity.HIGH if fact.is_critical else FailureSeverity.MEDIUM,
                            expected=f"Metric '{metric}' = {exp_val} (+/-{tol*100}%)",
                            actual="Value not found within tolerance interval",
                            evidence=f"Evidence pool: {agent_result.evidence[:2]}",
                            explanation=f"Agent analysis failed to substantiate ground truth fact: {fact.description}.",
                            recommended_fix="Verify statistical aggregation tool calculations.",
                        )
                    )
            else:
                # Qualitative fact check
                if fact.description.lower() in findings_str.lower():
                    verified_facts += 1

        factual_acc = round(verified_facts / len(facts), 4) if facts else 1.0

        # 2. Root Cause Accuracy
        acceptable = ground_truth.acceptable_causes
        root_cause_acc = 1.0
        if acceptable:
            matched_cause = False
            for acc in acceptable:
                if acc.cause.lower() in findings_str.lower():
                    matched_cause = True
                    break
            if not matched_cause:
                root_cause_acc = 0.5
                failures.append(
                    FailureRecord(
                        failure_id=f"FAIL-CAUSE-{ground_truth.scenario_id}",
                        scenario_id=ground_truth.scenario_id,
                        run_id=agent_result.run_id,
                        category=FailureCategory.F14_CONFLICT_RESOLUTION_ERROR,
                        severity=FailureSeverity.HIGH,
                        expected=f"Primary root cause in {[a.cause for a in acceptable]}",
                        actual="Identified root cause does not match acceptable set",
                        evidence=f"Agent findings: {agent_result.findings}",
                        explanation="Agent concluded an unsupported root cause.",
                        recommended_fix="Review multi-factor conflict resolution scoring weights.",
                    )
                )

        # 3. Recommendation Quality
        recs = agent_result.recommendations
        rec_themes = ground_truth.expected_recommendation_themes
        recs_str = " ".join([r.get("recommendation", "") for r in recs]).lower()

        matched_themes = sum(1 for theme in rec_themes if theme.lower() in recs_str)
        rec_rel = round(matched_themes / len(rec_themes), 4) if rec_themes else 1.0
        rec_act = 1.0 if recs else 0.5
        rec_ground = 1.0 if recs else 0.5

        overall_outcome = round(
            (0.35 * factual_acc) + (0.35 * root_cause_acc) + (0.30 * rec_rel),
            4,
        )

        outcome_metrics = OutcomeMetrics(
            analytical_accuracy=round((factual_acc + root_cause_acc) / 2.0, 4),
            numerical_accuracy=factual_acc,
            factual_accuracy=factual_acc,
            root_cause_accuracy=root_cause_acc,
            recommendation_relevance=rec_rel,
            recommendation_actionability=rec_act,
            recommendation_grounding=rec_ground,
            overall_outcome_score=overall_outcome,
        )

        return outcome_metrics, failures
