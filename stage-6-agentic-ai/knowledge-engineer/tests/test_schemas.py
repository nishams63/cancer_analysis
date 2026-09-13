"""Unit tests for Pydantic schema validation and business invariants."""
import pytest
from pydantic import ValidationError
from schemas.knowledge import (
    KnowledgeItem,
    KnowledgeCategory,
    RetrievalQuery,
    SearchResult,
    RetrievalResponse,
)


def test_valid_knowledge_item_creation(sample_item):
    assert sample_item.knowledge_id == "TEST-001"
    assert sample_item.category == KnowledgeCategory.DATA_CLEANING
    assert sample_item.status == "active"
    score = sample_item.compute_quality_score()
    assert 0.0 <= score <= 1.0
    assert score >= 0.8  # Well-populated item should score high


def test_invalid_knowledge_id_rejected():
    with pytest.raises(ValidationError):
        KnowledgeItem(
            knowledge_id="INVALIDIDNO_HYPHEN",
            title="Invalid ID Item",
            category=KnowledgeCategory.DATA_QUALITY,
            subcategory="test",
            description="Testing invalid ID formatting enforcement.",
        )


def test_invalid_status_rejected():
    with pytest.raises(ValidationError):
        KnowledgeItem(
            knowledge_id="DQ-999",
            title="Invalid Status Item",
            category=KnowledgeCategory.DATA_QUALITY,
            subcategory="test",
            description="Testing invalid status value enforcement.",
            status="invalid_status_enum",
        )


def test_short_title_rejected():
    with pytest.raises(ValidationError):
        KnowledgeItem(
            knowledge_id="DQ-998",
            title="AB",  # Too short (< 3 chars)
            category=KnowledgeCategory.DATA_QUALITY,
            subcategory="test",
            description="Testing short title rejection.",
        )


def test_searchable_text_generation(sample_item):
    text = sample_item.to_searchable_text()
    assert "Title: Test Imputation Method" in text
    assert "Category: data_cleaning" in text
    assert "Median imputation" in text
    assert "Underestimates standard error" in text


def test_retrieval_query_defaults():
    q = RetrievalQuery(query="How to impute?")
    assert q.top_k == 5
    assert q.status == "active"
    assert q.category is None
    assert q.min_score == 0.0


def test_retrieval_query_empty_rejected():
    with pytest.raises(ValidationError):
        RetrievalQuery(query="")
