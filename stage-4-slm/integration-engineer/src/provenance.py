"""
Provenance Tracking Module for Clinical SLM Inference.
Computes cryptographic hashes of inputs, prompts, outputs, and runtime state.
"""

import hashlib
from typing import Dict, Any


class ProvenanceTracker:
    """Calculates cryptographic hashes and bundles provenance metadata."""

    @staticmethod
    def hash_text(text: str) -> str:
        """Computes SHA-256 hash of a text string."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def build_provenance(
        inference_id: str,
        clinical_note: str,
        prompt: str,
        raw_output: str,
        model_name: str,
        adapter_version: str,
        quantization: str,
        prompt_version: str,
        tokenizer_version: str,
        runtime_version: str
    ) -> Dict[str, Any]:
        """Constructs provenance dictionary with hashes and version identifiers."""
        return {
            "inference_id": inference_id,
            "input_hash": ProvenanceTracker.hash_text(clinical_note),
            "prompt_hash": ProvenanceTracker.hash_text(prompt),
            "output_hash": ProvenanceTracker.hash_text(raw_output),
            "model_version": model_name,
            "adapter_version": adapter_version,
            "quantization": quantization,
            "prompt_version": prompt_version,
            "tokenizer_version": tokenizer_version,
            "runtime_version": runtime_version
        }
