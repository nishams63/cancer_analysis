import sys
from pathlib import Path
import pytest

SLM_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = SLM_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SLM_ROOT) not in sys.path:
    sys.path.insert(0, str(SLM_ROOT))
