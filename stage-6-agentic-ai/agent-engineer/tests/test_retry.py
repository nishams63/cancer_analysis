"""Tests for bounded exponential backoff execution."""
import pytest
from execution.retry import execute_with_retry


def test_retry_success_first_attempt():
    calls = 0
    def work():
        nonlocal calls
        calls += 1
        return "success_val"

    ok, val, attempts, err = execute_with_retry(work, max_retries=3, initial_delay_sec=0.01)
    assert ok is True
    assert val == "success_val"
    assert attempts == 1
    assert err is None


def test_retry_recovers_after_failure():
    calls = 0
    def flaky():
        nonlocal calls
        calls += 1
        if calls < 2:
            raise ConnectionError("Network blip")
        return "recovered"

    ok, val, attempts, err = execute_with_retry(flaky, max_retries=3, initial_delay_sec=0.01)
    assert ok is True
    assert val == "recovered"
    assert attempts == 2
    assert err is None


def test_retry_exhaustion():
    def always_fails():
        raise TimeoutError("Service timeout")

    ok, val, attempts, err = execute_with_retry(always_fails, max_retries=3, initial_delay_sec=0.01)
    assert ok is False
    assert val is None
    assert attempts == 3
    assert "Service timeout" in err
