"""Unit tests for LLM client and prompt construction."""
import pytest
from src.llm.nvidia_client import MockNvidiaLLMClient
from src.llm.prompt_builder import PromptBuilder
from src.llm.retry_handler import RetryHandler


def test_mock_nvidia_client_returns_structured_note():
    pb = PromptBuilder()
    client = MockNvidiaLLMClient()
    patient = {
        "patient_id": "SYN-P000001",
        "demographics": {"age": 62, "sex": "Female", "cancer_type": "NSCLC", "cancer_stage": "Stage IV"},
        "mutations": ["EGFR L858R", "EGFR T790M"],
        "biomarkers": {"tumor_size_cm": 2.8},
        "treatments": [{"treatment_name": "Osimertinib 80mg daily", "line": 2}],
        "dosages": {"Osimertinib": 80.0},
        "adverse_events": [],
        "resistance": {"status": True, "mechanism": "T790M"},
        "timeline": [{"event": "Diagnosis", "date": "2024-01-01", "delta_days": 0}],
        "missing_fields": []
    }
    scenario = {"scenario_id": "PROMPT-R01", "clinical_premise": "Test premise"}
    evidence = "CHUNK_ID: CHK-01 | Osimertinib therapy"

    messages = pb.build_messages(patient, scenario, evidence)
    resp = client.generate(messages)
    assert resp.content is not None
    assert "Osimertinib" in resp.content
    assert "EGFR" in resp.content
    assert resp.model == "meta/llama-3.1-70b-instruct-mock"


def test_prompt_builder_separates_facts_and_evidence():
    pb = PromptBuilder()
    patient = {
        "patient_id": "SYN-P000001",
        "demographics": {"age": 65, "sex": "Male", "cancer_type": "Melanoma", "cancer_stage": "Stage IV"},
        "mutations": ["BRAF V600E"],
        "biomarkers": {},
        "treatments": [{"treatment_name": "Pembrolizumab", "line": 1}],
        "dosages": {},
        "adverse_events": [],
        "resistance": {"status": False, "mechanism": None},
        "timeline": [],
        "missing_fields": ["lactate_dehydrogenase"]
    }
    scenario = {"scenario_id": "PROMPT-R11", "clinical_premise": "ICI re-challenge"}
    evidence = "CHUNK: NCCN Melanoma Guidelines"

    messages = pb.build_messages(patient, scenario, evidence)
    assert len(messages) == 2
    sys_p = messages[0]["content"]
    user_p = messages[1]["content"]
    assert "structured synthetic patient profile is the source of truth" in sys_p
    assert "You MUST NOT alter, remove, contradict" in sys_p
    assert "STRUCTURED SYNTHETIC PATIENT GROUND TRUTH" in user_p
    assert "BRAF V600E" in user_p
    assert "RETRIEVED ONCOLOGY EVIDENCE" in user_p


def test_retry_handler_success():
    rh = RetryHandler(max_attempts=2, backoff_factor=1.0)
    calls = 0

    def mock_fn():
        nonlocal calls
        calls += 1
        return "success"

    res = rh.execute(mock_fn)
    assert res == "success"
    assert calls == 1
