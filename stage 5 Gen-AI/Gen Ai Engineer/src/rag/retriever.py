"""Master RAG Retriever orchestrating search, filtering, and validation."""
import os
from typing import Dict, Any, List, Tuple
from .document_loader import DocumentLoader
from .vector_store import VectorStore
from .query_builder import RAGQueryBuilder
from .metadata_filter import MetadataFilter
from .reranker import Reranker
from .retrieval_validator import RetrievalValidator
from .evidence_formatter import EvidenceFormatter


class MasterRetriever:
    def __init__(self, index_path: str = None):
        self.index_path = index_path or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "retrieval", "indexes", "evidence_vector_index.json")
        self.doc_loader = DocumentLoader()
        self.vector_store = VectorStore()
        self.query_builder = RAGQueryBuilder()
        self.metadata_filter = MetadataFilter()
        self.reranker = Reranker()
        self.validator = RetrievalValidator()
        self.formatter = EvidenceFormatter()

        # Load or initialize vector store
        if os.path.exists(self.index_path):
            self.vector_store.load(self.index_path)
        else:
            chunks = self.doc_loader.load_all_chunks()
            if chunks:
                self.vector_store.build_index(chunks)
                self.vector_store.save(self.index_path)

    def retrieve_evidence(self, scenario: Dict[str, Any], patient: Dict[str, Any],
                          top_k: int = 5) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
        query_spec = self.query_builder.build_query(scenario, patient)
        raw_candidates = self.vector_store.search(query_spec["query_string"], top_k=top_k * 2)

        # Filter approved
        approved_chunks = self.metadata_filter.filter_chunks(raw_candidates, approved_only=True)

        # Rerank
        reranked = self.reranker.rerank(query_spec, approved_chunks)[:top_k]

        # Validate
        is_valid, violations, details = self.validator.validate_retrieval(query_spec, reranked, patient)

        # Format context string
        context_str = self.formatter.format_evidence_block(reranked)

        details["is_valid"] = is_valid
        details["violations"] = violations
        details["query_spec"] = query_spec
        return reranked, context_str, details