"""Random seed management and reproducibility context."""
import random
import numpy as np


class SeedContext:
    """Context manager to guarantee local deterministic reproducibility."""
    def __init__(self, seed: int):
        self.seed = seed
        self.prev_py_state = None
        self.prev_np_state = None

    def __enter__(self):
        self.prev_py_state = random.getstate()
        self.prev_np_state = np.random.get_state()
        random.seed(self.seed)
        np.random.seed(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        random.setstate(self.prev_py_state)
        np.random.set_state(self.prev_np_state)


def set_seed(seed: int = 42):
    """Globally set the random seed for Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)
