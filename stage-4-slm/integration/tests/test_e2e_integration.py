"""
End-to-End Integration Test for Stage 3 to Stage 4 Handoff.
Tests consumption of Stage 3 contracts and full decision support pipeline.
"""

import sys
import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

INTEG_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(INTEG_SRC_DIR))

from app import app

client = TestClient(app)


def test_stage3_to_stage4_e2e_handoff():
    """Simulates Stage 3 clinical payload processed by Stage 4 decision support API."""
    stage3_simulated_payload = {
        "document_id": "DOC-STG3-001",
        "patient_id": "PT-ONCO-888",
        "clinical_note": (
            "ONCOLOGY CONSULTATION PROGRESS NOTE\n"
            "Patient ID: PT-ONCO-888\n"
            "Primary Diagnosis: Stage IV Lung Adenocarcinoma with EGFR driver mutation.\n"
            "Plan: Continue Erlotinib 150 mg oral daily.\n"
            "Assessment: Tolerating therapy without acute toxicities. Labs stable."
        )
    }

    t0 = time.perf_counter()
    response = client.post("/v1/generate/decision-support", json=stage3_simulated_payload)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "DOC-STG3-001"
    assert data["patient_id"] == "PT-ONCO-888"
    assert "Erlotinib" in data["target_risk"]
    assert "Erlotinib" in data["target_key_finding"]
    assert "Continue standard" in data["target_action"]
    assert data["safety_gate_passed"] is True
    assert data["routing"] == "STAGE_4_AUTO"
    assert latency_ms < 250.0  # Well within hospital clinical SLA (<250ms)
