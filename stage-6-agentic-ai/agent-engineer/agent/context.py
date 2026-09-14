"""Task context builder providing compact, non-polluted execution frames."""
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from workflow_engineer.schemas.task import Task
from schemas.agent import AgentState


class TaskExecutionContext(BaseModel):
    """Focused task context supplied to ReAct step."""
    task_id: str
    task_name: str
    task_description: str
    allowed_tools: List[str]
    required_inputs: List[str]
    expected_outputs: List[str]
    previous_observations: Dict[str, Any]
    knowledge_context: str
    runtime_variables: Dict[str, Any]


def build_task_context(
    task: Task, state: AgentState, knowledge_context: str = "None"
) -> TaskExecutionContext:
    allowed_tools = [task.tool]
    if task.knowledge_retrieval:
        allowed_tools.append("retrieve_knowledge")

    return TaskExecutionContext(
        task_id=task.task_id,
        task_name=task.name,
        task_description=task.description,
        allowed_tools=allowed_tools,
        required_inputs=task.required_inputs,
        expected_outputs=task.expected_outputs,
        previous_observations=dict(state.observations),
        knowledge_context=knowledge_context,
        runtime_variables=dict(state.variables),
    )
