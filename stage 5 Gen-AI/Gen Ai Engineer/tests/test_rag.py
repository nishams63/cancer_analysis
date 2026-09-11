"""Unit tests for RAG retrieval on approved oncology evidence."""
import pytest
from src.rag import DocumentLoader, VectorStore, RAGQueryBuilder, MasterRetriever
from src.rag.evidence_formatter import EvidenceFormatter


def test_document_loader_approved_only():
    loader = DocumentLoader()
    docs = loader.load_all_chunks()
    assert len(docs) > 0
    for doc in docs:
        assert "chunk_id" in doc
        assert "text" in doc


def test_vector_store_indexing_and_search():
    loader = DocumentLoader()
    docs = loader.load_all_chunks()
    store = VectorStore()
    store.add_documents(docs)
    assert store.count() > 0

    results = store.similarity_search("Osimertinib resistance MET amplification", top_k=3)
    assert len(results) > 0
    assert "retrieval_score" in results[0]
    assert results[0]["retrieval_score"] >= results[-1]["retrieval_score"]


def test_query_builder_constructs_query():
    qb = RAGQueryBuilder()
    scenario = {
        "rag_intent": {
            "primary_query": "NCCN EGFR TKI resistance Osimertinib",
            "required_keywords": ["EGFR", "Osimertinib"]
        }
    }
    spec = qb.build_query(scenario, {})
    assert "Osimertinib" in spec["query_string"]
    assert spec["target_domain"] == "guidelines"


def test_evidence_formatter_grounding_block():
    formatter = EvidenceFormatter()
    chunks = [
        {
            "chunk_id": "CHK-001",
            "source": "NCCN_NSCLC_2024",
            "text": "Osimertinib is first line for EGFR exon 19 del or L858R.",
            "retrieval_score": 0.88
        }
    ]
    formatted = formatter.format_context(chunks)
    assert "CHUNK_ID: CHK-001" in formatted
    assert "Source: NCCN_NSCLC_2024" in formatted
    assert "Osimertinib is first line" in formatted


def test_master_retriever_full_cycle():
    retriever = MasterRetriever()
    scenario = {
        "scenario_id": "PROMPT-R01",
        "rag_intent": {
            "primary_query": "Osimertinib MET amplification resistance",
            "required_keywords": ["Osimertinib"]
        }
    }
    patient = {"demographics": {"cancer_type": "NSCLC"}}
    chunks, ctx, details = retriever.retrieve_evidence(scenario, patient, top_k=3)
    assert len(chunks) > 0
    assert details["is_valid"] is True
    assert "CHUNK_ID" in ctx
