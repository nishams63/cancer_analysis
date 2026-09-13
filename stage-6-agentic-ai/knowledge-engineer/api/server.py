"""FastAPI REST Service for Knowledge Base Queries and Management."""
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# Ensure parent directory is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schemas.knowledge import (
    KnowledgeItem,
    RetrievalQuery,
    RetrievalResponse,
    KnowledgeCategory,
)
from database.db_manager import DatabaseManager
from retrieval.vector_store import LocalVectorStore
from retrieval.ranker import HybridRanker
from retrieval.retriever import KnowledgeRetriever
from ingestion.ingest import run_ingestion_pipeline

app = FastAPI(
    title="AADA Knowledge Base Service",
    version="1.0.0",
    description="Queryable, versioned Knowledge Base API for Autonomous AI Data Analyst agents.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_manager = DatabaseManager()
vstore = LocalVectorStore()
vstore.load()
ranker = HybridRanker()
retriever = KnowledgeRetriever(db_manager=db_manager, vector_store=vstore, ranker=ranker)


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Check service health and database stats."""
    total_items = db_manager.count_items(status="active")
    return {
        "status": "healthy",
        "service": "AADA Knowledge Base",
        "total_active_items": total_items,
        "vector_store_indexed": vstore.is_indexed,
    }


@app.post("/knowledge/search", response_model=RetrievalResponse)
def search_knowledge(query_payload: RetrievalQuery) -> RetrievalResponse:
    """Search knowledge base using semantic + keyword + metadata hybrid retrieval."""
    return retriever.retrieve(query_payload)


@app.get("/knowledge/items/{knowledge_id}", response_model=KnowledgeItem)
def get_knowledge_item(knowledge_id: str) -> KnowledgeItem:
    """Retrieve single authoritative knowledge item by unique ID."""
    item = db_manager.get_item(knowledge_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Knowledge item '{knowledge_id}' not found.")
    return item


@app.get("/knowledge/categories")
def get_categories() -> List[Dict[str, Any]]:
    """List all domains and item counts."""
    return db_manager.get_categories()


@app.get("/knowledge/sources")
def get_sources() -> List[str]:
    """List all registered methodology sources."""
    return db_manager.get_sources()


@app.get("/knowledge/versions")
def get_versions() -> List[str]:
    """List all available knowledge versions."""
    return db_manager.get_versions()


@app.get("/knowledge/items")
def list_knowledge_items(
    category: Optional[str] = Query(None),
    subcategory: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    version: Optional[str] = Query(None),
    status: Optional[str] = Query("active"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> List[KnowledgeItem]:
    """List and filter raw knowledge items."""
    return db_manager.list_items(
        category=category,
        subcategory=subcategory,
        source=source,
        version=version,
        status=status,
        limit=limit,
        offset=offset,
    )


@app.post("/knowledge/ingest")
def trigger_ingestion(background_tasks: BackgroundTasks) -> Dict[str, str]:
    """Trigger ingestion pipeline from default knowledge directory."""
    kb_dir = PROJECT_ROOT / "knowledge_base"
    background_tasks.add_task(run_ingestion_pipeline, kb_dir)
    return {"status": "accepted", "message": "Ingestion pipeline triggered in background."}
