"""Automated root-cause diagnostic analyzer for agent execution failures."""
from typing import List, Dict, Any
from schemas.failure import FailureRecord, FailureCategory, FailureSeverity
from schemas.evaluation import ScenarioEvaluationResult


class ErrorAnalyzer:
    """Diagnoses root cause patterns from scenario evaluation defect records."""

    @staticmethod
    def analyze_failures(failures: List[FailureRecord]) -> Dict[str, Any]:
        critical_count = sum(1 for f in failures if f.severity == FailureSeverity.CRITICAL)
        high_count = sum(1 for f in failures if f.severity == FailureSeverity.HIGH)
        medium_count = sum(1 for f in failures if f.severity == FailureSeverity.MEDIUM)

        categories: Dict[str, int] = {}
        recommendations: List[str] = []

        for f in failures:
            cat = f.category.value if hasattr(f.category, "value") else str(f.category)
            categories[cat] = categories.get(cat, 0) + 1
            if f.recommended_fix and f.recommended_fix not in recommendations:
                recommendations.append(f.recommended_fix)

        return {
            "total_defects": len(failures),
            "critical_defects": critical_count,
            "high_defects": high_count,
            "medium_defects": medium_count,
            "category_distribution": categories,
            "prioritized_fixes": recommendations,
        }
