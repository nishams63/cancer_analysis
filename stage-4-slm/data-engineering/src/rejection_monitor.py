"""
Rejection-Rate Circuit Breaker Module for Stage 4.
Monitors batch and global rejection rates against configurable acceptance policies.
Halts execution if rejection rates spike beyond safety thresholds per Section 6b.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd

logger = logging.getLogger("stage4.rejection_monitor")


class CircuitBreakerError(Exception):
    """Raised when dataset or batch rejection rate exceeds safety thresholds."""
    pass


class RejectionMonitor:
    """Monitors rejection trends and enforces circuit-breaker policies."""

    def __init__(self, max_rejection_rate: float = 0.30, warn_rejection_rate: float = 0.20):
        self.max_rejection_rate = max_rejection_rate
        self.warn_rejection_rate = warn_rejection_rate
        self.trend_log: List[Dict[str, Any]] = []

    def check_circuit_breaker(
        self,
        total_count: int,
        reject_count: int,
        batch_id: str = "GLOBAL_DATASET",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculates rejection rate and checks against threshold.
        Raises CircuitBreakerError if rejection rate > max_rejection_rate.
        """
        if total_count == 0:
            rejection_rate = 0.0
        else:
            rejection_rate = reject_count / total_count

        status = "PASSED"
        if rejection_rate > self.max_rejection_rate:
            status = "HALTED"
        elif rejection_rate > self.warn_rejection_rate:
            status = "WARNING"

        record = {
            "batch_id": batch_id,
            "total_records": total_count,
            "rejected_records": reject_count,
            "rejection_rate": round(rejection_rate, 4),
            "threshold": self.max_rejection_rate,
            "status": status,
            "metadata": metadata or {}
        }
        self.trend_log.append(record)

        if status == "HALTED":
            err_msg = (
                f"CIRCUIT BREAKER TRIGGERED in batch '{batch_id}'! "
                f"Rejection rate {rejection_rate:.2%} exceeded configured threshold {self.max_rejection_rate:.2%}. "
                f"Pipeline execution halted to prevent silent dataset degradation."
            )
            logger.error(err_msg)
            raise CircuitBreakerError(err_msg)
        elif status == "WARNING":
            logger.warning(
                "Circuit breaker warning in batch '%s': Rejection rate %.2f%% exceeds warning threshold %.2f%%.",
                batch_id, rejection_rate * 100, self.warn_rejection_rate * 100
            )
        else:
            logger.info(
                "Circuit breaker passed for batch '%s': Rejection rate %.2f%% (threshold: %.2f%%).",
                batch_id, rejection_rate * 100, self.max_rejection_rate * 100
            )

        return record

    def check_subgroup_trends(self, df: pd.DataFrame, group_col: str = "document_type") -> List[Dict[str, Any]]:
        """Analyzes rejection rate trends across specific document types or sources."""
        if group_col not in df.columns or "entity_check_status" not in df.columns:
            return []

        subgroup_results = []
        for grp_val, group in df.groupby(group_col):
            tot = len(group)
            rej = int((group["entity_check_status"] != "PASS").sum())
            rate = round(rej / tot, 4) if tot > 0 else 0.0
            subgroup_results.append({
                "subgroup_dimension": group_col,
                "subgroup_value": str(grp_val),
                "total": tot,
                "rejected": rej,
                "rejection_rate": rate,
                "status": "HALT_RISK" if rate > self.max_rejection_rate else "NORMAL"
            })

        return subgroup_results
