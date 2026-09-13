"""Knowledge Retriever coordinating Vector Store, Database Metadata, and Hybrid Ranker."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from schemas.knowledge import (
    RetrievalQuery,
    RetrievalResponse,
    SearchResult,
    KnowledgeItem,
    KnowledgeCategory,
)
from database.db_manager import DatabaseManager
from retrieval.vector_store import LocalVectorStore
from retrieval.ranker import HybridRanker


class KnowledgeRetriever:
    """Unified high-level retrieval interface for Agent tools and API services."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        vector_store: Optional[LocalVectorStore] = None,
        ranker: Optional[HybridRanker] = None,
    ):
        self.db = db_manager or DatabaseManager()
        self.vector_store = vector_store or LocalVectorStore()
        self.ranker = ranker or HybridRanker()

    def retrieve(self, query_obj: RetrievalQuery) -> RetrievalResponse:
        """Execute hybrid search with metadata pre-filtering and relevance re-ranking."""
        start_time = time.perf_counter()

        # Step 1: Pre-filter candidate IDs from SQLite DB using strict metadata
        category_val = query_obj.category.value if isinstance(query_obj.category, KnowledgeCategory) else query_obj.category
        candidate_items = self.db.list_items(
            category=category_val,
            subcategory=query_obj.subcategory,
            source=query_obj.source,
            version=query_obj.version,
            status=query_obj.status,
            limit=200,
        )

        if not candidate_items:
            latency = (time.perf_counter() - start_time) * 1000.0
            return RetrievalResponse(
                query=query_obj.query,
                total_found=0,
                results=[],
                latency_ms=round(latency, 2),
                filters_applied={
                    "category": category_val,
                    "subcategory": query_obj.subcategory,
                    "source": query_obj.source,
                    "version": query_obj.version,
                    "status": query_obj.status,
                },
            )

        candidate_map: Dict[str, KnowledgeItem] = {item.knowledge_id: item for item in candidate_items}

        # Step 2: Semantic vector search
        vector_matches = self.vector_store.search(query_obj.query, top_k=50)
        semantic_scores: Dict[str, float] = {k_id: score for k_id, score in vector_matches}

        # Step 3: Rank candidates
        filtered_candidates = [
            candidate_map[kid] for kid in candidate_map.keys()
        ]

        ranked_results = self.ranker.rank(
            query=query_obj.query,
            items=filtered_candidates,
            semantic_scores=semantic_scores,
            top_k=query_obj.top_k,
            min_score=query_obj.min_score,
        )

        latency = (time.perf_counter() - start_time) * 1000.0
        return RetrievalResponse(
            query=query_obj.query,
            total_found=len(ranked_results),
            results=ranked_results,
            latency_ms=round(latency, 2),
            filters_applied={
                "category": category_val,
                "subcategory": query_obj.subcategory,
                "source": query_obj.source,
                "version": query_obj.version,
                "status": query_obj.status,
            },
        )
