"""
Production Version Rollback and Safety Circuit Breaker Module.
Supports Automatic, Recommended, and Manual rollback modes with strict safety safeguards,
human approval gating, and complete audit logging.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger("stage6_eval.rollback_manager")


class RollbackManager:
    """Manages active production deployment versions and executes controlled safety rollbacks."""

    VALID_MODES = {"automatic", "recommended", "manual"}

    def __init__(
        self,
        log_dir: str,
        active_version: str = "v2_filtered_r16",
        fallback_version: str = "v1_baseline",
        mode: str = "recommended",
        firewall_rejection_threshold: float = 0.05,
        require_human_approval: bool = True
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.log_dir / "rollback_history.json"

        self.active_version = active_version
        self.fallback_version = fallback_version
        self.mode = mode.lower() if mode.lower() in self.VALID_MODES else "recommended"
        self.firewall_rejection_threshold = firewall_rejection_threshold
        self.require_human_approval = require_human_approval

    def evaluate_health_and_decide(
        self,
        recent_monitoring_report: Dict[str, Any],
        approver_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluates monitoring metrics against thresholds.
        Executes or recommends rollback based on mode and approval policy.
        """
        rej_rate = recent_monitoring_report.get("rejection_rate", 0.0)
        has_alert = recent_monitoring_report.get("alert", False)

        threshold_breached = rej_rate > self.firewall_rejection_threshold or has_alert
        action_taken = "NO_ACTION_REQUIRED"

        if threshold_breached:
            if self.mode == "automatic":
                if self.require_human_approval and not approver_name:
                    action_taken = "ROLLBACK_RECOMMENDED_PENDING_APPROVAL"
                else:
                    action_taken = self._execute_rollback(
                        reason=f"Firewall rejection rate ({rej_rate*100:.1f}%) exceeded {self.firewall_rejection_threshold*100:.1f}%",
                        approver=approver_name or "AUTOMATED_SAFETY_DAEMON"
                    )
            elif self.mode == "recommended":
                action_taken = "ROLLBACK_RECOMMENDED_HUMAN_INTERVENTION_REQUIRED"
            else: # manual
                action_taken = "MANUAL_INVESTIGATION_FLAGGED"

        decision = {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "active_version": self.active_version,
            "fallback_version": self.fallback_version,
            "mode": self.mode,
            "threshold_breached": threshold_breached,
            "rejection_rate": rej_rate,
            "decision": action_taken,
            "require_human_approval": self.require_human_approval
        }

        return decision

    def _execute_rollback(self, reason: str, approver: str) -> str:
        """Performs state transition and logs to history."""
        old_active = self.active_version
        new_active = self.fallback_version

        entry = {
            "rollback_id": f"RBK_{int(datetime.utcnow().timestamp())}",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "from_version": old_active,
            "to_version": new_active,
            "trigger_reason": reason,
            "authorized_by": approver,
            "status": "SUCCESSFUL"
        }

        history = []
        if self.history_file.exists():
            try:
                history = json.loads(self.history_file.read_text(encoding="utf-8"))
            except Exception:
                history = []

        history.append(entry)
        self.history_file.write_text(json.dumps(history, indent=2), encoding="utf-8")

        self.active_version = new_active
        logger.warning(f"ROLLBACK EXECUTED: Reverted from {old_active} to {new_active}. Authorized by: {approver}")
        return f"ROLLBACK_EXECUTED_TO_{new_active}"
