"""
Autoregressive Inference Engine for Stage 5 SLM Engineering.
Uses genuine Hugging Face `model.generate()` to generate clinical outputs token-by-token.
Rules and parsers are strictly downstream validation/formatting, never answer synthesizers.
"""

import time
from typing import Dict, Any, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from .prompts import build_clinical_prompt, parse_clinical_output, DEFAULT_SYSTEM_PROMPT
from .utils import setup_logger

logger = setup_logger("inference")


class AutoregressiveInferenceEngine:
    """
    Executes real token-by-token autoregressive generation using Qwen2.5-1.5B.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.device = next(model.parameters()).device
        logger.info(f"Inference engine initialized on device: {self.device}")

    def generate(
        self,
        clinical_note: str,
        max_new_tokens: int = 128,
        temperature: float = 0.1,
        top_p: float = 0.95,
        do_sample: bool = False
    ) -> Dict[str, Any]:
        """
        Executes genuine autoregressive token generation.
        Returns the generated tokens, latency, and parsed fields.
        """
        prompt_str = build_clinical_prompt(
            self.tokenizer,
            clinical_note=clinical_note,
            system_prompt=self.system_prompt
        )

        inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.device)
        input_token_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        elapsed_sec = time.perf_counter() - t0
        latency_ms = elapsed_sec * 1000.0

        # Decode newly generated assistant tokens only
        generated_tokens = output_ids[0][input_token_len:]
        raw_completion = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
        output_token_count = len(generated_tokens)
        tokens_per_sec = output_token_count / max(0.001, elapsed_sec)

        # Parse generated text into clinical schema without overriding model predictions
        parsed_fields = parse_clinical_output(raw_completion)

        return {
            "prompt": prompt_str,
            "raw_completion": raw_completion,
            "parsed": parsed_fields,
            "input_tokens": input_token_len,
            "output_tokens": output_token_count,
            "latency_ms": round(latency_ms, 2),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "device": str(self.device),
            "generation_mode": "autoregressive_neural"
        }
