"""Metrics aggregation across experiment scenario suites."""
from typing import List, Dict
from schemas.evaluation import ScenarioEvaluationResult, OverallEvaluationResult


def aggregate_evaluation_results(
    evaluation_id: str,
    results: List[ScenarioEvaluationResult],
    timestamp: str,
    duration_seconds: float = 0.0,
) -> OverallEvaluationResult:
    """Aggregate a collection of scenario results into an overall evaluation report."""
    total = len(results)
    if total == 0:
        return OverallEvaluationResult(
            evaluation_id=evaluation_id,
            timestamp=timestamp,
            total_scenarios=0,
            passed_scenarios=0,
            failed_scenarios=0,
            success_rate=0.0,
            average_process_score=0.0,
            average_outcome_score=0.0,
            average_overall_score=0.0,
            scenario_results=[],
            failure_counts_by_category={},
            failure_counts_by_severity={},
            duration_seconds=duration_seconds,
        )

    passed_count = sum(1 for r in results if r.passed)
    failed_count = total - passed_count
    success_rate = round(passed_count / total, 4)

    avg_process = round(sum(r.process_score for r in results) / total, 4)
    avg_outcome = round(sum(r.outcome_score for r in results) / total, 4)
    avg_overall = round(sum(r.overall_score for r in results) / total, 4)

    cat_counts: Dict[str, int] = {}
    sev_counts: Dict[str, int] = {}

    for r in results:
        for f in r.failures:
            cat = f.category.value if hasattr(f.category, "value") else str(f.category)
            sev = f.severity.value if hasattr(f.severity, "value") else str(f.severity)
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            sev_counts[sev] = sev_counts.get(sev, 0) + 1

    return OverallEvaluationResult(
        evaluation_id=evaluation_id,
        timestamp=timestamp,
        total_scenarios=total,
        passed_scenarios=passed_count,
        failed_scenarios=failed_count,
        success_rate=success_rate,
        average_process_score=avg_process,
        average_outcome_score=avg_outcome,
        average_overall_score=avg_overall,
        scenario_results=results,
        failure_counts_by_category=cat_counts,
        failure_counts_by_severity=sev_counts,
        duration_seconds=duration_seconds,
    )
