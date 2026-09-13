"""Pytest fixtures and environment setup for Knowledge Engineer tests."""
import sys
import shutil
import tempfile
from pathlib import Path
import pytest

# Ensure package root is in sys.path
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from schemas.knowledge import KnowledgeItem, KnowledgeCategory
from database.db_manager import DatabaseManager
from retrieval.vector_store import LocalVectorStore
from retrieval.ranker import HybridRanker
from retrieval.retriever import KnowledgeRetriever


@pytest.fixture(scope="session")
def sample_item() -> KnowledgeItem:
    return KnowledgeItem(
        knowledge_id="TEST-001",
        title="Test Imputation Method",
        category=KnowledgeCategory.DATA_CLEANING,
        subcategory="imputation",
        description="A robust method for handling missing data in test scenarios.",
        conditions=["Missing values detected", "Sample size N > 50"],
        recommended_methods=["Median imputation", "K-nearest neighbors"],
        selection_rules=["Use median if skewed", "Use KNN if correlated"],
        limitations=["Underestimates standard error"],
        source="AADA Internal Methodology",
        version="1.0",
        effective_date="2026-01-01",
        status="active",
    )


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_kb.db"
    manager = DatabaseManager(db_path=db_file)
    return manager


@pytest.fixture
def temp_vector_store(tmp_path):
    index_dir = tmp_path / "vector_index"
    store = LocalVectorStore(index_dir=index_dir)
    return store


@pytest.fixture(scope="session")
def populated_retriever():
    """Retriever pointing to the authoritative database and vector store."""
    db = DatabaseManager()
    vstore = LocalVectorStore()
    vstore.load()
    ranker = HybridRanker()
    return KnowledgeRetriever(db_manager=db, vector_store=vstore, ranker=ranker)
