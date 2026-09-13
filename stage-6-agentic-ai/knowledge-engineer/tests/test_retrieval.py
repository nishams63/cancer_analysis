"""Unit tests for local vector similarity, keyword matching, and metadata filtering."""
import pytest
from schemas.knowledge import RetrievalQuery, KnowledgeCategory
from retrieval.vector_store import LocalVectorStore
from retrieval.ranker import HybridRanker
from retrieval.retriever import KnowledgeRetriever


def test_vector_store_fit_search(temp_vector_store, sample_item):
    temp_vector_store.build_index([sample_item])
    assert temp_vector_store.is_indexed
    
    matches = temp_vector_store.search("imputation of missing values", top_k=5)
    assert len(matches) == 1
    k_id, score = matches[0]
    assert k_id == "TEST-001"
    assert score > 0.1


def test_hybrid_ranker_keywords(sample_item):
    ranker = HybridRanker()
    score = ranker.compute_keyword_score("median imputation", sample_item)
    assert score > 0.2


def test_authoritative_retrieval_missing_values(populated_retriever):
    query = RetrievalQuery(
        query="How should I handle missing values in a highly skewed continuous variable?",
        top_k=5,
    )
    response = populated_retriever.retrieve(query)
    assert response.total_found > 0
    top_ids = [r.item.knowledge_id for r in response.results]
    # Should find numerical missing value assessment or imputation strategy
    assert "DC-001" in top_ids or "DQ-001" in top_ids


def test_authoritative_retrieval_isolation_forest(populated_retriever):
    query = RetrievalQuery(
        query="Multivariate anomaly detection using tree isolation path lengths",
        top_k=3,
    )
    response = populated_retriever.retrieve(query)
    assert response.total_found > 0
    assert response.results[0].item.knowledge_id == "ANOM-003"


def test_category_metadata_filtering(populated_retriever):
    # Search for regression strictly within visualization category
    query = RetrievalQuery(
        query="regression line overlay",
        category=KnowledgeCategory.VISUALIZATION,
        top_k=5,
    )
    response = populated_retriever.retrieve(query)
    for res in response.results:
        assert res.item.category == KnowledgeCategory.VISUALIZATION


def test_source_and_version_filtering(populated_retriever):
    query = RetrievalQuery(
        query="hypothesis testing",
        source="AADA Internal Methodology",
        version="1.0",
        top_k=5,
    )
    response = populated_retriever.retrieve(query)
    for res in response.results:
        assert res.item.source == "AADA Internal Methodology"
        assert res.item.version == "1.0"


def test_empty_query_handling(populated_retriever):
    query = RetrievalQuery(query="nonexistent_gibberish_term_xyz_12345", top_k=5)
    response = populated_retriever.retrieve(query)
    # Even if keyword/semantic is zero, should not crash
    assert isinstance(response.results, list)
