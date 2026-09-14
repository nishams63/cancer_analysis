"""Deterministic mock LLM provider for unit tests and offline execution."""
from typing import Dict, Any, List, Optional
from .base import LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """Deterministic LLM for testing without network dependency or token consumption."""

    def __init__(self, responses: Optional[Dict[str, LLMResponse]] = None):
        self.responses = responses or {}
        self.call_count = 0
        self.last_prompt = ""

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> LLMResponse:
        self.call_count += 1
        self.last_prompt = prompt

        if prompt in self.responses:
            return self.responses[prompt]

        # If tools provided, simulate selecting the most relevant one
        if tools and len(tools) > 0:
            selected_tool = tools[0]["function"]["name"]
            # Look for keyword matches in prompt
            for t in tools:
                name = t["function"]["name"]
                if name.replace("_", " ") in prompt.lower() or name in prompt.lower():
                    selected_tool = name
                    break

            return LLMResponse(
                action="tool_call",
                tool=selected_tool,
                arguments={},
                content=f"Selected permitted tool '{selected_tool}' for analytical task execution.",
                token_count=120,
            )

        # Default completion response
        return LLMResponse(
            action="final_response",
            tool=None,
            arguments={},
            content="Task completed successfully.",
            token_count=95,
        )
