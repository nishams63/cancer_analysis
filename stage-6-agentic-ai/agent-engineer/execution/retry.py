"""Bounded exponential backoff retry execution helper."""
import time
from typing import Callable, Any, Tuple, Optional


def execute_with_retry(
    fn: Callable[[], Any],
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    initial_delay_sec: float = 0.5,
) -> Tuple[bool, Any, int, Optional[str]]:
    """Execute callable with exponential backoff.
    
    Returns (success, result, attempts_made, error_msg).
    """
    delay = initial_delay_sec
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            res = fn()
            return True, res, attempt, None
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries:
                time.sleep(delay)
                delay *= backoff_factor

    return False, None, max_retries, last_err
