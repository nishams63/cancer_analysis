"""Abstract LLM provider interface and response contracts."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Standardized response contract for all LLM providers."""
    action: str = Field(..., description="Action classification: tool_call or final_response")
    tool: Optional[str] = Field(None, description="Selected tool name if tool_call")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Parsed arguments for tool")
    content: str = Field("", description="Natural language reasoning summary or message")
    token_count: int = Field(0, description="Tokens used by prompt and completion")
    raw_response: Dict[str, Any] = Field(default_factory=dict, description="Raw provider payload")


class LLMProvider(ABC):
    """Abstract base class for all language model providers in AADA."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a structured response or tool call decision."""
        pass
