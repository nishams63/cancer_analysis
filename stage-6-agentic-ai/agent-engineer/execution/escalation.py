"""Escalation rule evaluation and human review pausing."""
from typing import Optional, Tuple
from schemas.agent import AgentState
from schemas.escalation import EscalationAction, EscalationRule
from workflow_engineer.schemas.workflow import Workflow


def check_escalations(
    workflow: Workflow, state: AgentState
) -> Optional[Tuple[EscalationRule, str]]:
    """Evaluate workflow escalation rules. Returns first triggered (rule, message) or None."""
    context = dict(state.variables)
    # Inject aggregated metrics
    context["retry_count"] = state.metrics.retries
    context["quality_score"] = state.variables.get("quality_score", 1.0)
    context["root_cause_confidence"] = state.variables.get("root_cause_confidence", 1.0)
    context["cause_score_margin"] = state.variables.get("cause_score_margin", 1.0)

    for rule in workflow.escalation_rules:
        is_triggered, msg = rule.evaluate(context)
        if is_triggered and msg:
            return rule, msg

    return None
