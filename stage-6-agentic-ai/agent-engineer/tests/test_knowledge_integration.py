"""Tests for Knowledge Engineer tool contract integration."""
from tools.registry import get_default_registry


def test_knowledge_retrieval_tool():
    registry = get_default_registry()
    res = registry.execute("retrieve_knowledge", {"query": "revenue drop analysis methods", "top_k": 3})
    assert res["status"] == "success"
    assert "knowledge" in res
    assert len(res["knowledge"]) >= 1


def test_knowledge_retrieval_fallback():
    registry = get_default_registry()
    res = registry.execute("retrieve_knowledge", {"query": "rare non-existent subject xyz", "top_k": 2})
    assert res["status"] == "success"
    assert "knowledge" in res
