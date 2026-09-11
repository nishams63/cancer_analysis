"""Tests for RAG Retrieval Quality Evaluator."""
import pytest
from src.evaluation.rag_quality import RAGQualityEvaluator


def test_rag_retrieval_all_required_present(sample_scenario_def):
    evaluator = RAGQualityEvaluator()
    retrieval_entry = {
        "scenario_id": "PROMPT-R01",
        "results": [
            {
                "chunk_id": "CHUNK-001",
                "document_id": "DOC-NCCN-002",
                "approval_status": "approved",
                "score": 0.45,
                "text": "Osimertinib resistance with concurrent MET amplification requires dual inhibition."
            }
        ]
    }
    res = evaluator.evaluate_retrieval(retrieval_entry, sample_scenario_def)
    assert res["provenance_valid"] is True
    assert res["status"] == "PASS"
    assert res["concept_coverage_ratio"] >= 0.50


def test_rag_missing_provenance_fails():
    evaluator = RAGQualityEvaluator()
    retrieval_entry = {
        "scenario_id": "PROMPT-R01",
        "results": [
            {"score": 0.50, "text": "Some unprovenanced guideline text without IDs"}
        ]
    }
    res = evaluator.evaluate_retrieval(retrieval_entry)
    assert res["provenance_valid"] is False
    assert "R05" in res["failure_codes"]


def test_rag_unapproved_source_fails():
    evaluator = RAGQualityEvaluator()
    retrieval_entry = {
        "scenario_id": "PROMPT-R01",
        "results": [
            {
                "chunk_id": "CHUNK-999",
                "document_id": "UNAPPROVED-DOC",
                "approval_status": "rejected",
                "score": 0.30,
                "text": "Unapproved blog post"
            }
        ]
    }
    res = evaluator.evaluate_retrieval(retrieval_entry)
    assert "R04" in res["failure_codes"]
    assert res["status"] == "FAIL"
