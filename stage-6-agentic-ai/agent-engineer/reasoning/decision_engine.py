"""Post-Task Branch and Escalation Decision Engine."""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from schemas.agent import AgentState, RunStatus
from schemas.condition import BranchCondition, BranchAction
from schemas.escalation import EscalationRule, EscalationAction


class DecisionEngine:
    """Evaluates task branch conditions and global escalation rules against agent state."""

    @staticmethod
    def evaluate_task_branch(
        conditions: List[BranchCondition], context: Dict[str, Any]
    ) -> Tuple[Optional[BranchAction], Optional[str]]:
        """Evaluate conditions for a task. Returns (action, target_task_id)."""
        for cond in conditions:
            if cond.evaluate(context):
                return cond.action, cond.target_task_id
        return None, None

    @staticmethod
    def evaluate_escalations(
        rules: List[EscalationRule], context: Dict[str, Any]
    ) -> List[Tuple[EscalationRule, str]]:
        """Evaluate escalation rules. Returns list of (triggered_rule, formatted_message)."""
        triggered = []
        for rule in rules:
            is_trig, msg = rule.evaluate(context)
            if is_trig and msg:
                triggered.append((rule, msg))
        return triggered
