"""
Tests for Model Loading, Hardware Inspection, and LoRA Configuration.
"""

from model_loader import inspect_system_hardware, ModelLoader
from qlora_config import QLoRAConfigFactory


def test_hardware_inspection():
    hw = inspect_system_hardware()
    assert "device_name" in hw
    assert "ram_total_gb" in hw
    assert "disk_free_gb" in hw
    assert isinstance(hw["cuda_available"], bool)


def test_model_access_qwen():
    loader = ModelLoader("stage-4-slm/slm-engineer/configs/qwen.yaml")
    auth_info = loader.check_access_and_auth()
    assert auth_info["model_name"] == "qwen"
    assert auth_info["requires_auth"] is False
    assert auth_info["is_accessible"] is True


def test_lora_config_factory():
    cfg = QLoRAConfigFactory.get_lora_config(r=16, alpha=32, dropout=0.05)
    assert cfg["r"] == 16
    assert cfg["lora_alpha"] == 32
    assert cfg["lora_dropout"] == 0.05
    assert len(cfg["target_modules"]) == 7
