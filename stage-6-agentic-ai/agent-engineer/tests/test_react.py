"""Tests for ReAct loop execution and bounds."""
from tools.registry import get_default_registry
from llm.mock import MockLLMProvider
from reasoning.react_loop import ReActLoop


def test_react_step_success():
    registry = get_default_registry()
    llm = MockLLMProvider()
    loop = ReActLoop(tool_registry=registry, llm_provider=llm, max_steps=5)

    res = loop.execute_step(
        task_id="T001",
        task_name="Profile Data",
        task_description="Profile columns",
        allowed_tools=["profile_dataset"],
        required_inputs=[],
        previous_observations={},
    )
    assert res.status == "success"
    assert res.tool_name == "profile_dataset"
    assert res.decision == "continue"


def test_react_loop_terminates_on_success():
    registry = get_default_registry()
    llm = MockLLMProvider()
    loop = ReActLoop(tool_registry=registry, llm_provider=llm, max_steps=5)

    steps = loop.run_loop(
        task_id="T001",
        task_name="Load Data",
        task_description="Load dataset",
        allowed_tools=["load_dataset"],
        required_inputs=[],
        previous_observations={},
    )
    assert len(steps) == 1
    assert steps[0].status == "success"


def test_react_loop_terminates_on_max_steps():
    registry = get_default_registry()
    llm = MockLLMProvider()
    # Mock loop with custom max_steps
    loop = ReActLoop(tool_registry=registry, llm_provider=llm, max_steps=3)
    
    steps = loop.run_loop(
        task_id="T002",
        task_name="Failing Task",
        task_description="Simulated step",
        allowed_tools=["load_dataset"],
        required_inputs=[],
        previous_observations={},
        max_steps=3,
    )
    assert len(steps) <= 3
