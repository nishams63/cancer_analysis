"""LLM provider package for Agent Engineer."""
from .base import LLMProvider, LLMResponse
from .mock import MockLLMProvider
from .prompts import (
    AGENT_SYSTEM_PROMPT,
    TOOL_SELECTION_PROMPT,
    OBSERVATION_ANALYSIS_PROMPT,
    FINAL_SUMMARY_PROMPT,
)

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "MockLLMProvider",
    "AGENT_SYSTEM_PROMPT",
    "TOOL_SELECTION_PROMPT",
    "OBSERVATION_ANALYSIS_PROMPT",
    "FINAL_SUMMARY_PROMPT",
]

# Lazy import for NVIDIAProvider so environment lacking key does not fail at import
def get_nvidia_provider(**kwargs):
    from .nvidia import NVIDIAProvider
    return NVIDIAProvider(**kwargs)
