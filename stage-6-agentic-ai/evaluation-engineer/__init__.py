"""Evaluation Engineer Subsystem - AADA Stage 06."""
import sys
import types
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
STAGE6_DIR = CURRENT_DIR.parent
WF_DIR = STAGE6_DIR / "workflow-engineer"
KE_DIR = STAGE6_DIR / "knowledge-engineer"
AE_DIR = STAGE6_DIR / "agent-engineer"

for p in [str(KE_DIR), str(WF_DIR), str(AE_DIR), str(CURRENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Module Aliases
if "workflow_engineer" not in sys.modules and WF_DIR.exists():
    wf_pkg = types.ModuleType("workflow_engineer")
    wf_pkg.__path__ = [str(WF_DIR)]
    sys.modules["workflow_engineer"] = wf_pkg

if "knowledge_engineer" not in sys.modules and KE_DIR.exists():
    ke_pkg = types.ModuleType("knowledge_engineer")
    ke_pkg.__path__ = [str(KE_DIR)]
    sys.modules["knowledge_engineer"] = ke_pkg

if "agent_engineer" not in sys.modules and AE_DIR.exists():
    ae_pkg = types.ModuleType("agent_engineer")
    ae_pkg.__path__ = [str(AE_DIR)]
    sys.modules["agent_engineer"] = ae_pkg

# Unify schemas and tools paths
try:
    import schemas
    for d in [CURRENT_DIR, AE_DIR, WF_DIR, KE_DIR]:
        sp = str(d / "schemas")
        if (d / "schemas").exists() and sp not in schemas.__path__:
            schemas.__path__.append(sp)
except ImportError:
    pass
