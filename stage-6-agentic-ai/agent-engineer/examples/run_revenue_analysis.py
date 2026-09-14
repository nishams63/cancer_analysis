"""
End-to-End Revenue Drop Analysis Example.

Demonstrates executing a complete analytical workflow using the Agent Engineer.
Runs with MockLLMProvider deterministically by default (or NVIDIAProvider if API key is set).
"""

import os
import sys
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AADA.Example")

# Add paths to sys.path
CURRENT_DIR = Path(__file__).resolve().parent
AGENT_DIR = CURRENT_DIR.parent
STAGE6_DIR = AGENT_DIR.parent

sys.path.insert(0, str(STAGE6_DIR / "knowledge-engineer"))
sys.path.insert(0, str(STAGE6_DIR / "workflow-engineer"))
sys.path.insert(0, str(AGENT_DIR))

# Unify schema paths
import schemas
for d in [AGENT_DIR, STAGE6_DIR / "workflow-engineer", STAGE6_DIR / "knowledge-engineer"]:
    sp = str(d / "schemas")
    if (d / "schemas").exists() and sp not in schemas.__path__:
        schemas.__path__.append(sp)

import types
if "workflow_engineer" not in sys.modules:
    wf_pkg = types.ModuleType("workflow_engineer")
    wf_pkg.__path__ = [str(STAGE6_DIR / "workflow-engineer")]
    sys.modules["workflow_engineer"] = wf_pkg

if "knowledge_engineer" not in sys.modules:
    ke_pkg = types.ModuleType("knowledge_engineer")
    ke_pkg.__path__ = [str(STAGE6_DIR / "knowledge-engineer")]
    sys.modules["knowledge_engineer"] = ke_pkg

from schemas.agent import RunStatus
from llm.mock import MockLLMProvider
from llm.nvidia import NVIDIAProvider
from tools.registry import get_default_registry
from trace.repository import TraceRepository
from agent.executor import AgentExecutor
from workflow_engineer.workflows.registry import WorkflowRegistry


def main():
    logger.info("=== AADA Agent Engineer: Running Revenue Drop Analysis Demo ===")
    
    # 1. Select LLM Provider
    api_key = os.environ.get("NVIDIA_API_KEY")
    if api_key:
        logger.info("NVIDIA_API_KEY detected. Initializing NVIDIAProvider...")
        llm = NVIDIAProvider(api_key=api_key)
    else:
        logger.info("No NVIDIA_API_KEY found. Initializing MockLLMProvider (deterministic offline mode)...")
        llm = MockLLMProvider()
    
    # 2. Tool Registry
    registry = get_default_registry()
    logger.info(f"Loaded {len(registry.list_tools())} analytical tools into registry.")
    
    # 3. Trace Repository
    db_path = CURRENT_DIR / "revenue_analysis_trace.db"
    if db_path.exists():
        try:
            db_path.unlink() # Fresh run
        except Exception:
            pass
    trace_repo = TraceRepository(db_path=str(db_path))
    
    # 4. Instantiate Agent Executor
    executor = AgentExecutor(
        llm_provider=llm,
        tool_registry=registry,
        trace_repository=trace_repo,
        max_react_steps=5
    )
    
    # 5. Load Workflow Template from Workflow Engineer
    wf_registry = WorkflowRegistry()
    workflow = wf_registry.get_template_for_intent("revenue_decline")
    if not workflow:
        workflow = wf_registry.get_template("WF-REVENUE-001")
    
    logger.info(f"Loaded workflow '{workflow.workflow_id}': '{workflow.name}' with {len(workflow.tasks)} tasks.")
    
    # 6. Execute Workflow
    logger.info("Executing workflow...")
    result = executor.run(workflow)
    
    # 7. Print Execution Summary
    print("\n" + "=" * 65)
    print("            AADA AGENT WORKFLOW EXECUTION SUMMARY")
    print("=" * 65)
    print(f"Run ID:             {result.run_id}")
    print(f"Workflow ID:        {result.workflow_id}")
    print(f"Final Status:       {result.status.upper()}")
    print(f"Confidence Score:   {result.confidence:.2%}")
    print(f"Execution Duration: {result.metrics.get('duration_seconds', 0.0):.3f}s")
    print(f"Total Tool Calls:   {result.metrics.get('tool_calls', 0)}")
    print(f"Total Tasks:        {len(result.completed_tasks)} completed")
    print("-" * 65)
    print("Completed Tasks:")
    for tid in result.completed_tasks:
        task_obj = workflow.get_task(tid)
        t_name = task_obj.name if task_obj else "Unknown"
        t_tool = task_obj.tool if task_obj else "Unknown"
        print(f"  [x] {tid:6} | Task: {t_name:25} | Tool: {t_tool}")
    
    if result.findings:
        print("-" * 65)
        print("Ranked Analytical Findings:")
        for idx, f in enumerate(result.findings, 1):
            cause = f.get("cause", "Finding")
            score = f.get("score", 0.0)
            print(f"  {idx}. {cause} (Score: {score})")

    if result.recommendations:
        print("-" * 65)
        print("Strategic Action Plan:")
        for idx, r in enumerate(result.recommendations, 1):
            rec = r.get("recommendation", str(r))
            print(f"  {idx}. {rec}")
    
    # 8. Verify Trace Persistence
    events = trace_repo.get_trace(result.run_id)
    print("=" * 65)
    print(f"Trace events recorded to SQLite ({db_path.name}): {len(events)} events.")
    print("=" * 65 + "\n")
    
    return result


if __name__ == "__main__":
    main()
