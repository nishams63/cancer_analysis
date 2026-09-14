"""Evaluation report generator producing structured JSON and formatted Markdown."""
import json
from pathlib import Path
from typing import Dict, Any, Optional
from schemas.evaluation import OverallEvaluationResult, ScenarioEvaluationResult


class ReportGenerator:
    """Compiles evaluation metrics into executive JSON and Markdown artifacts."""

    @staticmethod
    def generate_json(result: OverallEvaluationResult) -> str:
        """Serialize overall evaluation result to pretty formatted JSON."""
        return result.model_dump_json(indent=2)

    @staticmethod
    def generate_markdown(result: OverallEvaluationResult) -> str:
        """Render publication-grade human-readable markdown evaluation report."""
        lines = []
        lines.append("============================================================")
        lines.append("                 AADA AGENT EVALUATION REPORT                ")
        lines.append("============================================================")
        lines.append(f"Evaluation ID:        {result.evaluation_id}")
        lines.append(f"Timestamp:            {result.timestamp}")
        lines.append(f"Scenarios Evaluated:  {result.total_scenarios}")
        lines.append(f"Successful (Passed):  {result.passed_scenarios}")
        lines.append(f"Failed:               {result.failed_scenarios}")
        lines.append(f"Overall Success Rate: {result.success_rate:.1%}")
        lines.append("------------------------------------------------------------")
        lines.append("")
        lines.append("## PROCESS METRICS")
        lines.append(f"- Average Process Score:        {result.average_process_score:.1%}")
        lines.append("")
        lines.append("## OUTCOME METRICS")
        lines.append(f"- Average Outcome Score:        {result.average_outcome_score:.1%}")
        lines.append(f"- Average Overall Score:        {result.average_overall_score:.1%}")
        lines.append("")
        lines.append("## SAFETY & CRITICAL INVARIANTS")
        crit = result.failure_counts_by_severity.get("critical", 0)
        high = result.failure_counts_by_severity.get("high", 0)
        lines.append(f"- Critical Safety Violations:   {crit}")
        lines.append(f"- High Severity Defects:        {high}")
        lines.append("")
        lines.append("## FAILURE CATALOG SUMMARY")
        if result.failure_counts_by_category:
            for cat, cnt in sorted(result.failure_counts_by_category.items()):
                lines.append(f"- {cat}: {cnt}")
        else:
            lines.append("- Zero defects recorded.")
        lines.append("")
        lines.append("## SCENARIO BREAKDOWN")
        lines.append("| Scenario ID | Run ID | Process | Outcome | Overall | Status | Verdict |")
        lines.append("|---|---|---|---|---|---|---|")
        for sc in result.scenario_results:
            status_badge = "PASS" if sc.passed else "FAIL"
            lines.append(
                f"| {sc.scenario_id} | {sc.run_id} | {sc.process_score:.1%} | {sc.outcome_score:.1%} | "
                f"{sc.overall_score:.1%} | **{status_badge}** | {sc.verdict_rationale[:40]} |"
            )
        lines.append("")
        lines.append("============================================================")
        return "\n".join(lines)

    @classmethod
    def save_reports(
        cls,
        result: OverallEvaluationResult,
        output_dir: Optional[Path | str] = None,
    ) -> Dict[str, str]:
        out = Path(output_dir or Path(__file__).resolve().parent)
        out.mkdir(parents=True, exist_ok=True)

        json_path = out / "evaluation_report.json"
        md_path = out / "evaluation_report.md"

        with open(json_path, "w", encoding="utf-8") as f:
            f.write(cls.generate_json(result))

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(cls.generate_markdown(result))

        return {"json": str(json_path), "markdown": str(md_path)}
