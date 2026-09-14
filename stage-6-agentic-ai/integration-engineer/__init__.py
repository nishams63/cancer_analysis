"""AADA Integration Engineer package."""
import sys
import types
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
STAGE6_DIR = ROOT_DIR.parent
WF_DIR = STAGE6_DIR / "workflow-engineer"
KE_DIR = STAGE6_DIR / "knowledge-engineer"
AGENT_DIR = STAGE6_DIR / "agent-engineer"
EVAL_DIR = STAGE6_DIR / "evaluation-engineer"

for p in [KE_DIR, WF_DIR, AGENT_DIR, EVAL_DIR, ROOT_DIR]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Register package aliases for hyphenated directory names
aliases = {
    "workflow_engineer": WF_DIR,
    "knowledge_engineer": KE_DIR,
    "agent_engineer": AGENT_DIR,
    "evaluation_engineer": EVAL_DIR,
    "integration_engineer": ROOT_DIR,
}
for name, path in aliases.items():
    if name not in sys.modules and path.exists():
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules[name] = pkg

# Multi-directory schemas unification
import schemas
for d in [ROOT_DIR, EVAL_DIR, AGENT_DIR, WF_DIR, KE_DIR]:
    sp = str(d / "schemas")
    if (d / "schemas").exists() and sp not in schemas.__path__:
        schemas.__path__.append(sp)

# Multi-directory tools unification
import tools
for d in [AGENT_DIR, KE_DIR]:
    tp = str(d / "tools")
    if (d / "tools").exists() and tp not in tools.__path__:
        tools.__path__.append(tp)
