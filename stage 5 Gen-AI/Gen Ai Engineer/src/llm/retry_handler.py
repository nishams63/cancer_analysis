"""Retry handler with exponential backoff for LLM API calls."""
import time
import random
from typing import Callable, Any


class RetryHandler:
    def __init__(self, max_attempts: int = 3, backoff_factor: float = 1.5, timeout_seconds: float = 30.0):
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self.timeout_seconds = timeout_seconds

    def execute(self, func: Callable[[], Any]) -> Any:
        attempts = 0
        last_exception = None

        while attempts < self.max_attempts:
            try:
                attempts += 1
                return func()
            except Exception as e:
                last_exception = e
                if attempts >= self.max_attempts:
                    break
                sleep_time = (self.backoff_factor ** attempts) + random.uniform(0.1, 0.5)
                time.sleep(sleep_time)

        raise RuntimeError(f"LLM API failed after {self.max_attempts} attempts: {last_exception}") from last_exception