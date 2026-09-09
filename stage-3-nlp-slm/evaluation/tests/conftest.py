"""
Pytest configuration for Stage 3 Clinical NLP Evaluation.
Ensures evaluation/src is prioritized in sys.path and isolates config from parent scopes.
"""

import sys
from pathlib import Path
import pytest

TEST_DIR = Path(__file__).resolve().parent
EVAL_SRC_DIR = (TEST_DIR.parent / "src").resolve()
NLP_SRC_DIR = (TEST_DIR.parent.parent / "nlp" / "src").resolve()

# Ensure evaluation/src is at the very front of sys.path
if str(EVAL_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_SRC_DIR))
else:
    sys.path.remove(str(EVAL_SRC_DIR))
    sys.path.insert(0, str(EVAL_SRC_DIR))

# Ensure nlp/src is also on sys.path for feature transformers
if str(NLP_SRC_DIR) not in sys.path:
    sys.path.append(str(NLP_SRC_DIR))

# Ensure 'config' in sys.modules points to evaluation/src/config.py
if "config" in sys.modules:
    del sys.modules["config"]
import config
