"""
Tests scanning the Stage 5 codebase to ensure zero simulation, zero fake metrics,
and zero fake loss generators remain in the primary engineering pipeline.
"""

from pathlib import Path


def test_no_simulation_in_src():
    """Scans all Python files in src/ to verify no simulation or placeholder logic exists."""
    src_dir = Path("stage-4-slm/slm-engineer/src")
    forbidden_tokens = [
        "LORA_ADAPTER_WEIGHTS_BIN_PLACEHOLDER",
        "random.normal",
        "simulate_training",
        "fake_eval",
        "simulated_loss"
    ]

    for py_file in src_dir.glob("**/*.py"):
        content = py_file.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in content, (
                f"Forbidden simulation token '{token}' found in {py_file.name}!"
            )


def test_inference_uses_model_generate():
    """Verifies that inference engine uses genuine model.generate() and not regex text generation."""
    inf_file = Path("stage-4-slm/slm-engineer/src/inference.py")
    content = inf_file.read_text(encoding="utf-8")

    assert "self.model.generate" in content, "InferenceEngine must call self.model.generate!"
    assert "re.search" not in content.split("def generate")[1].split("parsed_fields =")[0], (
        "InferenceEngine.generate() must not synthesize output text using regex!"
    )
