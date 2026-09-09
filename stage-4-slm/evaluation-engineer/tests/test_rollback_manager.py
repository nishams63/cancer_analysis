"""
Unit tests for Rollback Manager and Production Safety Controller.
"""

import tempfile
from pathlib import Path
import pytest
from rollback_manager import RollbackManager


def test_rollback_manager_no_action_when_healthy():
    with tempfile.TemporaryDirectory() as tmp_dir:
        mgr = RollbackManager(
            log_dir=tmp_dir,
            active_version="v2_active",
            fallback_version="v1_baseline",
            mode="recommended",
            firewall_rejection_threshold=0.05
        )

        healthy_report = {"rejection_rate": 0.02, "alert": False}
        decision = mgr.evaluate_health_and_decide(healthy_report)
        assert decision["threshold_breached"] is False
        assert decision["decision"] == "NO_ACTION_REQUIRED"
        assert mgr.active_version == "v2_active"


def test_rollback_manager_recommended_mode_flags_intervention():
    with tempfile.TemporaryDirectory() as tmp_dir:
        mgr = RollbackManager(
            log_dir=tmp_dir,
            active_version="v2_active",
            fallback_version="v1_baseline",
            mode="recommended",
            firewall_rejection_threshold=0.05
        )

        unhealthy_report = {"rejection_rate": 0.08, "alert": True}
        decision = mgr.evaluate_health_and_decide(unhealthy_report)
        assert decision["threshold_breached"] is True
        assert decision["decision"] == "ROLLBACK_RECOMMENDED_HUMAN_INTERVENTION_REQUIRED"
        assert mgr.active_version == "v2_active" # Did not auto-execute without approval


def test_rollback_manager_automatic_execution():
    with tempfile.TemporaryDirectory() as tmp_dir:
        mgr = RollbackManager(
            log_dir=tmp_dir,
            active_version="v2_active",
            fallback_version="v1_baseline",
            mode="automatic",
            firewall_rejection_threshold=0.05,
            require_human_approval=False
        )

        unhealthy_report = {"rejection_rate": 0.09, "alert": True}
        decision = mgr.evaluate_health_and_decide(unhealthy_report)
        assert decision["threshold_breached"] is True
        assert "ROLLBACK_EXECUTED_TO_v1_baseline" in decision["decision"]
        assert mgr.active_version == "v1_baseline"
        assert (Path(tmp_dir) / "rollback_history.json").exists()
