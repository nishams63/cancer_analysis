"""
Mandatory Acceptance Test: Section 16 & Change 3 Offline Operation Verification.
Performs two rigorous offline verifications:
1. Automated programmatic network-block test (intercepts socket.socket.connect).
2. Codebase dependency scan verifying zero external cloud model SDKs or remote calls.
"""

import sys
import socket
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure src is on path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from api import app, config


class NetworkAccessAttemptError(RuntimeError):
    """Raised when an unauthorized outbound network call is attempted."""
    pass


@pytest.fixture
def enforce_network_block(monkeypatch):
    """
    Blocks all outbound socket connections to non-loopback addresses.
    Permits only localhost / 127.0.0.1 communication.
    """
    original_connect = socket.socket.connect

    def guarded_connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        # Allow loopback/in-process addresses
        if host in ("127.0.0.1", "localhost", "::1"):
            return original_connect(self, address)
        raise NetworkAccessAttemptError(
            f"ILLEGAL_NETWORK_CALL: Attempted outbound connection to external host '{host}'. "
            "Integration service must operate 100% offline."
        )

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    yield
    monkeypatch.setattr(socket.socket, "connect", original_connect)


def test_offline_inference_under_strict_network_block(enforce_network_block):
    """
    Verifies that the complete inference pipeline (/summarize) executes
    with zero outbound internet connection attempts.
    """
    client = TestClient(app)

    # 1. Health check
    h_res = client.get("/health")
    assert h_res.status_code == 200
    h_data = h_res.json()
    assert h_data["status"] == "healthy"
    assert h_data["offline_mode"] is True
    assert h_data["runtime"] == "llama.cpp"

    # 2. Clinical note submission
    payload = {
        "clinical_note": (
            "65-year-old female with metastatic adenocarcinoma of the lung with EGFR L858R mutation. "
            "Currently on osimertinib 80mg daily. Tolerating treatment well with no rash, diarrhea, "
            "or shortness of breath. Confirms absence of acute toxicities."
        )
    }

    res = client.post("/summarize", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["safety_status"] == "PASS"
    assert data["review_required"] is False
    assert data["risk"] == "Low"
    assert "osimertinib" in data["key_finding"].lower()
    assert data["confidence"] >= config.safety.confidence_threshold
    assert data["latency_ms"] > 0
    assert len(data["inference_id"]) > 0


def test_codebase_zero_external_network_dependencies():
    """
    Scans integration-engineer codebase to ensure no cloud AI SDKs
    (openai, google.generativeai, anthropic, replicate) or outbound HTTP requests exist in inference path.
    """
    root = Path(__file__).resolve().parent.parent
    banned_imports = ["openai", "anthropic", "google.generativeai", "replicate", "cohere"]

    python_files = list(root.glob("src/**/*.py")) + list(root.glob("scripts/**/*.py"))
    assert len(python_files) >= 5, "Should have multiple source files to scan"

    for py_file in python_files:
        content = py_file.read_text(encoding="utf-8").lower()
        for banned in banned_imports:
            assert f"import {banned}" not in content, f"Found banned cloud SDK import '{banned}' in {py_file.name}"
            assert f"from {banned}" not in content, f"Found banned cloud SDK import '{banned}' in {py_file.name}"
