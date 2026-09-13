"""Unit tests for Agent Tool interface: retrieve_knowledge."""
from tools.retrieve_knowledge import retrieve_knowledge, RETRIEVE_KNOWLEDGE_TOOL_SPEC


def test_retrieve_knowledge_success():
    res = retrieve_knowledge(
        query="How to detect outliers without assuming normal distribution?",
        category="anomaly_detection",
        top_k=3,
    )
    assert res["status"] == "success"
    assert res["count"] > 0
    assert "knowledge" in res
    assert len(res["knowledge"]) <= 3
    
    first = res["knowledge"][0]
    assert "knowledge_id" in first
    assert "title" in first
    assert "relevance_score" in first
    assert "recommended_methods" in first
    assert first["category"] == "anomaly_detection"


def test_retrieve_knowledge_empty_query_error():
    res = retrieve_knowledge(query="   ")
    assert res["status"] == "error"
    assert res["count"] == 0
    assert "must not be empty" in res["message"]


def test_tool_specification_contract():
    spec = RETRIEVE_KNOWLEDGE_TOOL_SPEC
    assert spec["name"] == "retrieve_knowledge"
    assert "parameters" in spec
    assert "query" in spec["parameters"]["properties"]
    assert "category" in spec["parameters"]["properties"]
    assert "required" in spec["parameters"]
    assert "query" in spec["parameters"]["required"]
    assert len(spec["parameters"]["properties"]["category"]["enum"]) == 9
