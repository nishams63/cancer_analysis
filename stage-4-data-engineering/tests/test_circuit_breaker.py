"""
Unit Tests for Section 6b Rejection-Rate Circuit Breaker.
Tests passing within tolerance and halting execution when rejection spikes.
"""

import pytest
from pathlib import Path
import sys

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from rejection_monitor import RejectionMonitor, CircuitBreakerError


def test_circuit_breaker_pass_below_threshold():
    """Test that a low rejection rate allows the pipeline to continue."""
    monitor = RejectionMonitor(max_rejection_rate=0.30, warn_rejection_rate=0.20)
    res = monitor.check_circuit_breaker(total_count=100, reject_count=10, batch_id="TEST_PASS")
    assert res["status"] == "PASSED"
    assert res["rejection_rate"] == 0.10


def test_circuit_breaker_warning():
    """Test that an intermediate rejection rate triggers a WARNING without halting."""
    monitor = RejectionMonitor(max_rejection_rate=0.30, warn_rejection_rate=0.20)
    res = monitor.check_circuit_breaker(total_count=100, reject_count=25, batch_id="TEST_WARN")
    assert res["status"] == "WARNING"
    assert res["rejection_rate"] == 0.25


def test_circuit_breaker_halts_above_threshold():
    """Test that exceeding the max rejection threshold raises CircuitBreakerError."""
    monitor = RejectionMonitor(max_rejection_rate=0.30)
    with pytest.raises(CircuitBreakerError) as exc_info:
        monitor.check_circuit_breaker(total_count=100, reject_count=35, batch_id="TEST_SPIKE")
    assert "CIRCUIT BREAKER TRIGGERED" in str(exc_info.value)
