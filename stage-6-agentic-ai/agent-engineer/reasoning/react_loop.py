"""ReAct reasoning loop (Thought -> Action -> Observation)."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from tools.registry import ToolRegistry
from llm.base import LLMProvider, LLMResponse
from llm.prompts import TOOL_SELECTION_PROMPT


class ReActStepResult(BaseModel):
    """Encapsulates a single step in the ReAct loop."""
    task_id: str
    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: Dict[str, Any]
    observation_summary: str
    decision: str
    decision_reason: str
    status: str = "success"
    step_number: int = 1


class ReActLoop:
    """Orchestrates structured Thought -> Tool Call -> Observation execution."""

    def __init__(self, tool_registry: ToolRegistry, llm_provider: LLMProvider, max_steps: int = 5):
        self.registry = tool_registry
        self.llm = llm_provider
        self.max_steps = max_steps

    def execute_step(
        self,
        task_id: str,
        task_name: str,
        task_description: str,
        allowed_tools: List[str],
        required_inputs: List[str],
        previous_observations: Dict[str, Any],
        knowledge_context: str = "None",
        explicit_inputs: Optional[Dict[str, Any]] = None,
        step_number: int = 1,
    ) -> ReActStepResult:
        """Execute a single ReAct analytical step."""
        tool_specs = self.registry.to_openai_specs(allowed_tools=allowed_tools)
        obs_summary = "\n".join([f"- {k}: {str(v)[:120]}" for k, v in previous_observations.items()]) or "None"

        prompt = TOOL_SELECTION_PROMPT.format(
            task_id=task_id,
            task_name=task_name,
            task_description=task_description,
            allowed_tools=", ".join(allowed_tools),
            required_inputs=", ".join(required_inputs) or "None",
            previous_observations=obs_summary,
            knowledge_context=knowledge_context,
        )

        llm_resp: LLMResponse = self.llm.generate(prompt=prompt, tools=tool_specs)

        selected_tool = llm_resp.tool
        if not selected_tool or selected_tool not in allowed_tools:
            selected_tool = allowed_tools[0]

        tool_args = dict(explicit_inputs or {})
        if llm_resp.arguments:
            tool_args.update(llm_resp.arguments)

        tool_output = self.registry.execute(
            tool_name=selected_tool,
            inputs=tool_args,
            allowed_tools=allowed_tools,
        )

        decision = "continue"
        reason = llm_resp.content or f"Executed tool '{selected_tool}' in accordance with task contract."

        return ReActStepResult(
            task_id=task_id,
            tool_name=selected_tool,
            tool_input=tool_args,
            tool_output=tool_output,
            observation_summary=f"Completed {selected_tool} with status '{tool_output.get('status', 'success')}'.",
            decision=decision,
            decision_reason=reason,
            status=tool_output.get("status", "success"),
            step_number=step_number,
        )

    def run_loop(
        self,
        task_id: str,
        task_name: str,
        task_description: str,
        allowed_tools: List[str],
        required_inputs: List[str],
        previous_observations: Dict[str, Any],
        knowledge_context: str = "None",
        explicit_inputs: Optional[Dict[str, Any]] = None,
        max_steps: Optional[int] = None,
    ) -> List[ReActStepResult]:
        """Run multi-step ReAct loop until task succeeds or max_steps reached."""
        limit = max_steps or self.max_steps
        steps: List[ReActStepResult] = []

        for step_idx in range(1, limit + 1):
            res = self.execute_step(
                task_id=task_id,
                task_name=task_name,
                task_description=task_description,
                allowed_tools=allowed_tools,
                required_inputs=required_inputs,
                previous_observations=previous_observations,
                knowledge_context=knowledge_context,
                explicit_inputs=explicit_inputs,
                step_number=step_idx,
            )
            steps.append(res)
            if res.status == "success":
                break

        return steps
