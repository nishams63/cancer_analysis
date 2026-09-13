"""Agent Tool Contract: retrieve_knowledge."""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# Ensure parent directory is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schemas.knowledge import RetrievalQuery, KnowledgeCategory
from database.db_manager import DatabaseManager
from retrieval.vector_store import LocalVectorStore
from retrieval.ranker import HybridRanker
from retrieval.retriever import KnowledgeRetriever

# Lazy-loaded singleton retriever
_GLOBAL_RETRIEVER: Optional[KnowledgeRetriever] = None


def get_retriever() -> KnowledgeRetriever:
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        db = DatabaseManager()
        vstore = LocalVectorStore()
        vstore.load()
        ranker = HybridRanker()
        _GLOBAL_RETRIEVER = KnowledgeRetriever(db_manager=db, vector_store=vstore, ranker=ranker)
    return _GLOBAL_RETRIEVER


def retrieve_knowledge(
    query: str,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    source: Optional[str] = None,
    version: Optional[str] = None,
    top_k: int = 5,
    min_score: float = 0.0,
) -> Dict[str, Any]:
    """Authoritative Agent Tool for accessing domain knowledge in the AADA system.
    
    Args:
        query: Natural language query, question, or intent (e.g. 'How should I handle missing skewed data?').
        category: Optional domain category filter (e.g. 'data_quality', 'statistics', 'eda').
        subcategory: Optional subcategory filter (e.g. 'imputation', 'hypothesis_testing').
        source: Optional source citation filter.
        version: Optional version filter.
        top_k: Number of ranked results to return (default 5).
        min_score: Minimum relevance threshold (default 0.0).

    Returns:
        Structured dictionary containing matched knowledge units, recommendations, and limitations.
    """
    if not query or not query.strip():
        return {
            "status": "error",
            "message": "Query string must not be empty.",
            "results": [],
            "count": 0,
        }

    retriever = get_retriever()
    cat_enum = None
    if category:
        try:
            cat_enum = KnowledgeCategory(category.strip().lower())
        except ValueError:
            pass  # Fall back to None or string filter

    query_obj = RetrievalQuery(
        query=query.strip(),
        category=cat_enum,
        subcategory=subcategory.strip() if subcategory else None,
        source=source.strip() if source else None,
        version=version.strip() if version else None,
        top_k=top_k,
        min_score=min_score,
    )

    response = retriever.retrieve(query_obj)

    formatted_items: List[Dict[str, Any]] = []
    for res in response.results:
        item = res.item
        formatted_items.append({
            "knowledge_id": item.knowledge_id,
            "title": item.title,
            "category": item.category.value,
            "subcategory": item.subcategory,
            "relevance_score": res.score,
            "semantic_similarity": res.semantic_score,
            "keyword_overlap": res.keyword_score,
            "description": item.description,
            "conditions": item.conditions,
            "recommended_methods": item.recommended_methods,
            "selection_rules": item.selection_rules,
            "limitations": item.limitations,
            "source": item.source,
            "version": item.version,
        })

    return {
        "status": "success",
        "query": query,
        "count": len(formatted_items),
        "latency_ms": response.latency_ms,
        "filters_applied": response.filters_applied,
        "knowledge": formatted_items,
    }


# Standard Tool Schema for LLM Function Calling
RETRIEVE_KNOWLEDGE_TOOL_SPEC = {
    "name": "retrieve_knowledge",
    "description": (
        "Retrieve authoritative data analysis methodology, cleaning rules, statistical tests, "
        "and machine learning guidelines from the AADA Knowledge Base."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language analytics intent or question (e.g. 'Handling high cardinality features').",
            },
            "category": {
                "type": "string",
                "enum": [
                    "data_quality",
                    "data_cleaning",
                    "eda",
                    "statistics",
                    "machine_learning",
                    "anomaly_detection",
                    "visualization",
                    "root_cause",
                    "business_analysis",
                ],
                "description": "Filter by primary analytics domain.",
            },
            "subcategory": {
                "type": "string",
                "description": "Filter by sub-domain (e.g. 'imputation', 'outliers', 'correlation').",
            },
            "source": {
                "type": "string",
                "description": "Filter by authoritative reference standard.",
            },
            "version": {
                "type": "string",
                "description": "Filter by knowledge version (e.g. '1.0').",
            },
            "top_k": {
                "type": "integer",
                "default": 5,
                "description": "Number of top ranked items to return.",
            },
        },
        "required": ["query"],
    },
}
