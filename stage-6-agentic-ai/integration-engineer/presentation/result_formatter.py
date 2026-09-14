"""Formats raw AgentResult into clean, evidence-grounded user presentation."""
from typing import Dict, Any, List, Optional
from agent_engineer.schemas.result import AgentResult


class ResultFormatter:
    """Transforms AgentResult into structured presentation models without fabricating data."""

    @staticmethod
    def classify_evidence(evidence_dict: Dict[str, Any]) -> str:
        """Classify evidence items as OBSERVED FACT or EVIDENCE-SUPPORTED INFERENCE."""
        # Simple rule-based classification based on direct metric values vs derived hypothesis
        if any(k in str(evidence_dict).lower() for k in ["rate", "margin", "decline", "magnitude", "total", "count", "revenue", "churn"]):
            return "OBSERVED FACT"
        return "EVIDENCE-SUPPORTED INFERENCE"

    @classmethod
    def format(cls, result: Optional[AgentResult], goal: str = "") -> Dict[str, Any]:
        if not result:
            return {
                "summary": "Analysis has not completed or no result available.",
                "findings": [],
                "evidence": [],
                "recommendations": [],
                "confidence": None,
                "limitations": "Execution was interrupted or pending human review.",
            }

        findings_list = []
        for f in result.findings:
            if isinstance(f, dict):
                findings_list.append({
                    "cause": f.get("cause") or f.get("finding") or "Primary factor",
                    "score": f.get("score"),
                    "details": f.get("details", ""),
                })
            else:
                findings_list.append({"cause": str(f), "score": None, "details": ""})

        classified_evidence = []
        for e in result.evidence:
            if isinstance(e, dict):
                classified_evidence.append({
                    "data": e,
                    "classification": cls.classify_evidence(e),
                })
            else:
                classified_evidence.append({
                    "data": {"value": str(e)},
                    "classification": "OBSERVED FACT",
                })

        recommendations_list = []
        for r in result.recommendations:
            if isinstance(r, dict):
                recommendations_list.append(r.get("recommendation", str(r)))
            else:
                recommendations_list.append(str(r))

        # Synthesize concise executive summary from findings
        top_cause = findings_list[0]["cause"] if findings_list else "Analysis completed"
        summary = f"Analysis for '{goal or result.objective}' concluded with confidence {int((result.confidence or 0.0) * 100)}%. Identified primary contributing factor: {top_cause}."

        return {
            "summary": summary,
            "findings": findings_list,
            "evidence": classified_evidence,
            "recommendations": recommendations_list,
            "confidence": result.confidence,
            "status": result.status,
            "completed_tasks_count": len(result.completed_tasks),
            "limitations": "Findings are based strictly on available execution data and allowable tools.",
        }
