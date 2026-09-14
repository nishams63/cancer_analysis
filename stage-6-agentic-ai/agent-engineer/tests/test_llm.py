"""Tests for LLM providers (MockLLMProvider and NVIDIAProvider)."""
import os
import pytest
from llm.mock import MockLLMProvider
from llm.nvidia import NVIDIAProvider
from llm.base import LLMResponse


def test_mock_llm_generation():
    llm = MockLLMProvider()
    tools = [{"type": "function", "function": {"name": "eda_analysis"}}]
    resp = llm.generate("Please perform EDA on sales dataset", tools=tools)
    assert isinstance(resp, LLMResponse)
    assert resp.tool == "eda_analysis"
    assert resp.token_count > 0


def test_mock_llm_tool_selection():
    llm = MockLLMProvider()
    tools = [{"type": "function", "function": {"name": "profile_dataset"}}]
    resp = llm.generate("Profile dataset columns", tools=tools)
    assert resp.tool == "profile_dataset"


def test_nvidia_provider_init_without_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        NVIDIAProvider()


@pytest.mark.skipif("NVIDIA_API_KEY" not in os.environ, reason="NVIDIA_API_KEY not set")
def test_real_nvidia_provider_call():
    provider = NVIDIAProvider()
    resp = provider.generate("Analyze why revenue dropped in Q3")
    assert resp.content
    assert resp.token_count > 0
