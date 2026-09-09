"""
Pytest configuration for Stage 3 NLP test suite.
Ensures local nlp/src directory takes precedence in sys.path and sys.modules.
"""

import sys
from pathlib import Path

NLP_SRC = Path(__file__).resolve().parent.parent / "src"
if str(NLP_SRC) not in sys.path:
    sys.path.insert(0, str(NLP_SRC))

# Ensure nlp/src/config.py is loaded, not any other config from preceding test suites
if "config" in sys.modules:
    cfg_file = getattr(sys.modules["config"], "__file__", "")
    if str(NLP_SRC) not in cfg_file:
        del sys.modules["config"]
