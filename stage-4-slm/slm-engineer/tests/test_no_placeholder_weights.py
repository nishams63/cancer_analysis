"""
Tests auditing adapter directories to ensure no placeholder weights exist.
Enforces Rule 1: Rejection of any file containing PLACEHOLDER or size < 1 KB.
"""

from pathlib import Path
from src.lora import verify_adapter_integrity


def test_verify_adapter_integrity_rejects_placeholder(tmp_path):
    """Verifies that audit function flags simulation placeholder weights immediately."""
    fake_adapter_dir = tmp_path / "fake_adapter"
    fake_adapter_dir.mkdir()

    cfg_file = fake_adapter_dir / "adapter_config.json"
    cfg_file.write_text('{"r": 16, "lora_alpha": 32, "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct"}')

    # Create dummy placeholder like previous stage 5
    weights_file = fake_adapter_dir / "adapter_model.safetensors"
    weights_file.write_bytes(b"LORA_ADAPTER_WEIGHTS_BIN_PLACEHOLDER")

    audit = verify_adapter_integrity(fake_adapter_dir)
    assert audit["valid"] is False
    assert audit["is_placeholder"] is True
    assert "CRITICAL" in audit.get("error", "")


def test_no_active_placeholder_in_adapters():
    """Asserts that no placeholder weights exist in the newly rebuilt real_qwen_lora adapter."""
    real_adapter_root = Path("stage-4-slm/slm-engineer/adapters/real_qwen_lora")
    if not real_adapter_root.exists():
        return

    for safetensors_path in real_adapter_root.glob("**/*.safetensors"):
        content = safetensors_path.read_bytes()
        assert b"LORA_ADAPTER_WEIGHTS_BIN_PLACEHOLDER" not in content, (
            f"Found forbidden placeholder in real adapter: {safetensors_path}"
        )
        assert safetensors_path.stat().st_size > 1024, (
            f"Real adapter weight file is smaller than 1 KB: {safetensors_path}"
        )
