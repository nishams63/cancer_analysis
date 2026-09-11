"""Base LLM Client interface and response models."""
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    content: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    raw_response: Dict[str, Any] = Field(default_factory=dict)


class LLMClient(ABC):
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate response from chat messages."""
        pass