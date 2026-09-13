"""Unit tests for FastAPI endpoints."""
from fastapi.testclient import TestClient
from api.server import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["total_active_items"] >= 50
    assert data["vector_store_indexed"] is True


def test_search_endpoint():
    payload = {
        "query": "Handling high cardinality categorical variables",
        "category": "data_cleaning",
        "top_k": 3,
    }
    response = client.post("/knowledge/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == payload["query"]
    assert data["total_found"] > 0
    assert len(data["results"]) <= 3
    assert data["results"][0]["item"]["category"] == "data_cleaning"


def test_get_item_by_id_found():
    response = client.get("/knowledge/items/DQ-001")
    assert response.status_code == 200
    data = response.json()
    assert data["knowledge_id"] == "DQ-001"
    assert data["title"] == "Missing Numerical Values Assessment"


def test_get_item_by_id_not_found():
    response = client.get("/knowledge/items/DOES-NOT-EXIST")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_categories_endpoint():
    response = client.get("/knowledge/categories")
    assert response.status_code == 200
    categories = response.json()
    assert len(categories) >= 8
    cat_names = [c["category"] for c in categories]
    assert "data_quality" in cat_names
    assert "statistics" in cat_names
    assert "machine_learning" in cat_names


def test_sources_endpoint():
    response = client.get("/knowledge/sources")
    assert response.status_code == 200
    sources = response.json()
    assert len(sources) > 0
    assert "AADA Internal Methodology" in sources


def test_versions_endpoint():
    response = client.get("/knowledge/versions")
    assert response.status_code == 200
    versions = response.json()
    assert "1.0" in versions


def test_list_items_endpoint():
    response = client.get("/knowledge/items?limit=10&offset=0")
    assert response.status_code == 200
    items = response.json()
    assert len(items) == 10
