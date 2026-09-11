import time
from contextlib import contextmanager
from typing import Generator, Dict, Any, Optional

class Stopwatch:
    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0
        self.elapsed_ms = 0.0

    def start(self):
        self.start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000.0
        return self.elapsed_ms

@contextmanager
def timed_step(step_name: str, metrics_dict: Optional[Dict[str, Any]] = None) -> Generator[Stopwatch, None, None]:
    sw = Stopwatch()
    sw.start()
    try:
        yield sw
    finally:
        sw.stop()
        if metrics_dict is not None:
            metrics_dict[f'{step_name}_latency_ms'] = round(sw.elapsed_ms, 2)
