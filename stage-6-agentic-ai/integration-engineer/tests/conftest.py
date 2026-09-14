"""Pytest configuration and shared fixtures for Integration Engineer."""
import sys
import types
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

TESTS_DIR = Path(__file__).resolve().parent
INTEG_DIR = TESTS_DIR.parent
STAGE6_DIR = INTEG_DIR.parent
WF_DIR = STAGE6_DIR / "workflow-engineer"
KE_DIR = STAGE6_DIR / "knowledge-engineer"
AGENT_DIR = STAGE6_DIR / "agent-engineer"
EVAL_DIR = STAGE6_DIR / "evaluation-engineer"

for p in [KE_DIR, WF_DIR, AGENT_DIR, EVAL_DIR, INTEG_DIR]:
    sp = str(p)
    if sp in sys.path:
        sys.path.remove(sp)
    sys.path.insert(0, sp)

# Register package aliases for hyphenated directory names
aliases = {
    "workflow_engineer": WF_DIR,
    "knowledge_engineer": KE_DIR,
    "agent_engineer": AGENT_DIR,
    "evaluation_engineer": EVAL_DIR,
    "integration_engineer": INTEG_DIR,
}
for name, path in aliases.items():
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    sys.modules[name] = pkg

# Multi-directory schemas unification
import schemas
for d in [INTEG_DIR, EVAL_DIR, AGENT_DIR, WF_DIR, KE_DIR]:
    sp = str(d / "schemas")
    if (d / "schemas").exists() and sp not in schemas.__path__:
        schemas.__path__.append(sp)

# Multi-directory tools unification
import tools
for d in [AGENT_DIR, KE_DIR]:
    tp = str(d / "tools")
    if (d / "tools").exists() and tp not in tools.__path__:
        tools.__path__.append(tp)

# Multi-directory api unification with INTEG_DIR prioritized
import api
for d in [INTEG_DIR, AGENT_DIR]:
    ap = str(d / "api")
    if (d / "api").exists() and ap not in api.__path__:
        api.__path__.insert(0, ap)

from api.server import app
from orchestration.pipeline import AADAIntegrationPipeline
from orchestration.session_manager import get_session_manager
from events.event_bus import get_global_event_bus


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def pipeline():
    return AADAIntegrationPipeline()


@pytest.fixture
def session_manager():
    mgr = get_session_manager()
    mgr.clear()
    return mgr


@pytest.fixture
def event_bus():
    bus = get_global_event_bus()
    bus.clear()
    return bus
