"""NVIDIA LLM API Client and local mock realization fallback."""
import os
import time
import requests
from typing import List, Dict, Any, Optional
from .base_client import LLMClient, LLMResponse
from .retry_handler import RetryHandler


class MockNvidiaLLMClient(LLMClient):
    """High-fidelity local deterministic realizing client used when API key is unset or in tests."""
    def __init__(self, model_name: str = "meta/llama-3.1-70b-instruct-mock"):
        self.model_name = model_name

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        t0 = time.time()
        user_msg = messages[-1]["content"] if messages else ""

        # Extract structured patient facts from user prompt
        import re
        age_m = re.search(r"- Age: ([0-9]+)", user_msg)
        age = age_m.group(1) if age_m else "62"

        sex_m = re.search(r"- Sex: (Male|Female)", user_msg)
        sex = sex_m.group(1) if sex_m else "Female"

        dx_m = re.search(r"- Primary Diagnosis: ([^\n]+)", user_msg)
        dx = dx_m.group(1) if dx_m else "NSCLC (Stage IV)"

        mut_m = re.search(r"- Confirmed Mutations: ([^\n]+)", user_msg)
        muts = mut_m.group(1) if mut_m else "EGFR T790M, MET Amplification"

        tx_m = re.search(r"- Current Treatment: ([^\n]+)", user_msg)
        tx = tx_m.group(1) if tx_m else "Osimertinib 80mg daily"

        # Synthesize realistic SOAP narrative respecting all extracted facts
        narrative = (
            f"# OUTPATIENT ONCOLOGY CLINICAL PROGRESS NOTE\n"
            f"**Patient Age**: {age} | **Sex**: {sex} | **Diagnosis**: {dx}\n"
            f"**Evaluation**: Synthetic scenario generated for oncology AI stress testing.\n\n"
            f"### SUBJECTIVE\n"
            f"The patient is a {age}-year-old {sex.lower()} with a history of {dx}. "
            f"Currently receiving {tx}. Patient is followed closely for disease response and tolerance.\n\n"
            f"### OBJECTIVE\n"
            f"- Genomic Testing / ctDNA: Confirmed {muts}.\n"
            f"- Current Antineoplastic Therapy: {tx}.\n"
            f"- Laboratory / Imaging Review: All objective parameters documented per clinical protocol.\n\n"
            f"### ASSESSMENT\n"
            f"{age}yo {sex.lower()} with {dx}. Genomic profile demonstrates {muts}. "
            f"Clinical findings and molecular profile are evaluated in accordance with approved guidelines.\n\n"
            f"### PLAN\n"
            f"1. Continue therapy as indicated under protocol guidelines: {tx}.\n"
            f"2. Maintain strict serial laboratory and restaging surveillance.\n"
            f"3. Return to clinic in 3 weeks or sooner if acute symptoms arise."
        )

        dt = (time.time() - t0) * 1000
        return LLMResponse(
            content=narrative,
            model=self.model_name,
            provider="mock_nvidia",
            prompt_tokens=len(user_msg) // 4,
            completion_tokens=len(narrative) // 4,
            total_tokens=(len(user_msg) + len(narrative)) // 4,
            latency_ms=round(dt, 2),
            raw_response={"mock": True}
        )


class NvidiaLLMClient(LLMClient):
    """Production client connecting to NVIDIA AI Foundation chat completions endpoint."""
    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None,
                 model: Optional[str] = None, retry_handler: Optional[RetryHandler] = None):
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self.api_base = api_base or os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.model = model or os.environ.get("LLM_MODEL", "meta/llama-3.1-70b-instruct")
        self.retry_handler = retry_handler or RetryHandler()

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        url = f"{self.api_base.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.2),
            "max_tokens": kwargs.get("max_tokens", 1200),
            "top_p": kwargs.get("top_p", 0.95)
        }

        t0 = time.time()
        def _call():
            resp = requests.post(url, json=payload, headers=headers, timeout=self.retry_handler.timeout_seconds)
            resp.raise_for_status()
            return resp.json()

        data = self.retry_handler.execute(_call)
        dt = (time.time() - t0) * 1000

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            provider="nvidia",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=round(dt, 2),
            raw_response=data
        )


def get_llm_client(config: Dict[str, Any] = None) -> LLMClient:
    """Factory creating NvidiaLLMClient if API key is present, otherwise MockNvidiaLLMClient."""
    try:
        from dotenv import load_dotenv
        for env_f in [os.path.join("stage5", ".env"), ".env"]:
            if os.path.exists(env_f):
                load_dotenv(env_f)
    except ImportError:
        pass

    api_key = os.environ.get("NVIDIA_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "meta/llama-3.2-11b-vision-instruct")
    api_base = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

    if config:
        m_cfg = config.get("model", {})
        if isinstance(m_cfg, dict):
            model = m_cfg.get("name", model)
            api_base = m_cfg.get("api_base", api_base)

    if api_key and not api_key.startswith("mock"):
        return NvidiaLLMClient(api_key=api_key, api_base=api_base, model=model)
    return MockNvidiaLLMClient()