"""Pytest fixtures and configuration for Stage 5 Gen AI Engineer."""
import sys
from pathlib import Path
import pytest

ROLE_ROOT = Path(__file__).resolve().parent.parent
STAGE5_ROOT = ROLE_ROOT.parent
WORKSPACE_ROOT = STAGE5_ROOT.parent
SRC_DIR = ROLE_ROOT / "src"

for p in [str(ROLE_ROOT), str(SRC_DIR), str(STAGE5_ROOT), str(WORKSPACE_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)
