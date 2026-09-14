"""Safe task executor coordinating tool allowlist and ReAct loop."""
from typing import Dict, Any, List, Optional
from schemas.agent import AgentState
from schemas.trace import EventType
from workflow_engineer.schemas.task import Task
from tools.registry import ToolRegistry
from llm.base import LLMProvider
from reasoning.react_loop import ReActLoop, ReActStepResult


class TaskExecutor:
    """Executes single workflow tasks within allowlisted boundaries."""

    def __init__(self, registry: ToolRegistry, llm: LLMProvider):
        self.registry = registry
        self.react_loop = ReActLoop(tool_registry=registry, llm_provider=llm)

    def execute_task(
        self,
        task: Task,
        state: AgentState,
        knowledge_context: str = "None",
        explicit_inputs: Optional[Dict[str, Any]] = None,
    ) -> ReActStepResult:
        """Execute analytical task enforcing tool allowlisting."""
        # Allowed tools: task.tool plus retrieve_knowledge if specified
        allowed_tools = [task.tool]
        if task.knowledge_retrieval:
            allowed_tools.append("retrieve_knowledge")

        step_res = self.react_loop.execute_step(
            task_id=task.task_id,
            task_name=task.name,
            task_description=task.description,
            allowed_tools=allowed_tools,
            required_inputs=task.required_inputs,
            previous_observations=state.observations,
            knowledge_context=knowledge_context,
            explicit_inputs=explicit_inputs,
        )

        state.metrics.tool_calls += 1
        return step_res
