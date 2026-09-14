"""Agent Engineer Subsystem - AADA Stage 06."""
import sys
import types
from pathlib import Path

# Setup paths
CURRENT_DIR = Path(__file__).resolve().parent
STAGE6_DIR = CURRENT_DIR.parent
WF_DIR = STAGE6_DIR / "workflow-engineer"
KE_DIR = STAGE6_DIR / "knowledge-engineer"

# Register workflow_engineer module if not present
if "workflow_engineer" not in sys.modules and WF_DIR.exists():
    wf_pkg = types.ModuleType("workflow_engineer")
    wf_pkg.__path__ = [str(WF_DIR)]
    sys.modules["workflow_engineer"] = wf_pkg

# Register knowledge_engineer module if not present
if "knowledge_engineer" not in sys.modules and KE_DIR.exists():
    ke_pkg = types.ModuleType("knowledge_engineer")
    ke_pkg.__path__ = [str(KE_DIR)]
    sys.modules["knowledge_engineer"] = ke_pkg

# Unify schemas and tools paths across Stage 6 subpackages
try:
    import schemas
    for d in [CURRENT_DIR, WF_DIR, KE_DIR]:
        sp = str(d / "schemas")
        if (d / "schemas").exists() and sp not in schemas.__path__:
            schemas.__path__.append(sp)
except ImportError:
    pass

try:
    import tools
    for d in [CURRENT_DIR, KE_DIR]:
        tp = str(d / "tools")
        if (d / "tools").exists() and tp not in tools.__path__:
            tools.__path__.append(tp)
except ImportError:
    pass
