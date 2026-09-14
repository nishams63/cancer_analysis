"""NVIDIA API LLM Provider with function calling and environment configuration."""
from __future__ import annotations
import os
import json
import time
from typing import Dict, Any, List, Optional
import httpx
from .base import LLMProvider, LLMResponse
from .prompts import AGENT_SYSTEM_PROMPT


class NVIDIAProvider(LLMProvider):
    """NVIDIA Cloud Function calling provider loading credentials safely from environment."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "NVIDIA_API_KEY environment variable is not set. "
                "Configure it via os.environ['NVIDIA_API_KEY'] before calling NVIDIAProvider."
            )
        self.model = model or os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
        self.base_url = (base_url or os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")).rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        **kwargs,
    ) -> LLMResponse:
        messages = [
            {"role": "system", "content": system_prompt or AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        retries = 3
        last_error = None

        for attempt in range(retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                    )

                if resp.status_code == 429:
                    time.sleep(1.0 * (attempt + 1))
                    continue

                resp.raise_for_status()
                data = resp.json()

                choice = data["choices"][0]
                message = choice["message"]

                # Check if model produced tool calls
                if "tool_calls" in message and message["tool_calls"]:
                    tool_call = message["tool_calls"][0]
                    tool_name = tool_call["function"]["name"]
                    try:
                        args = json.loads(tool_call["function"]["arguments"])
                    except Exception:
                        args = {}
                    return LLMResponse(
                        action="tool_call",
                        tool=tool_name,
                        arguments=args,
                        content=message.get("content") or "",
                        raw_response=data,
                    )
                else:
                    return LLMResponse(
                        action="final_response",
                        tool=None,
                        arguments={},
                        content=message.get("content") or "",
                        raw_response=data,
                    )

            except Exception as e:
                last_error = e
                time.sleep(0.5 * (attempt + 1))

        raise RuntimeError(f"NVIDIA API call failed after {retries} attempts: {last_error}")
