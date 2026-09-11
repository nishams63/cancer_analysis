"""Lightweight local persistent vector store for oncology evidence chunks."""
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from .embedding_service import EmbeddingService


class VectorStore:
    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

    def build_index(self, chunks: List[Dict[str, Any]]) -> int:
        """Index a list of chunk dictionaries."""
        self.chunks = chunks
        texts = [f"{c.get('section', '')} {c.get('text', '')}" for c in chunks]
        self.embeddings = self.embedding_service.fit_transform(texts)
        return len(chunks)

    def search(self, query: str, top_k: int = 5, min_score: float = 0.1) -> List[Dict[str, Any]]:
        """Search top-k most similar chunks using cosine similarity."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        q_vec = self.embedding_service.transform([query])
        # Cosine similarity is dot product of normalized vectors
        scores = np.dot(self.embeddings, q_vec.T).flatten()
        top_indices = np.argsort(-scores)[:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= min_score:
                chunk_copy = dict(self.chunks[idx])
                chunk_copy["retrieval_score"] = round(score, 4)
                results.append(chunk_copy)
        return results

    # Convenience aliases
    add_documents = build_index
    similarity_search = search
    def count(self) -> int:
        return len(self.chunks)

    def save(self, filepath: str):
        """Persist index chunks and vocabulary to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        index_data = {
            "version": "1.0.0",
            "provider": self.embedding_service.provider,
            "dimension": self.embedding_service.dimension,
            "total_chunks": len(self.chunks),
            "chunks": self.chunks,
            "vocabulary": list(self.embedding_service.vectorizer.vocabulary_.keys()) if self.embedding_service.is_fitted else []
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

    def load(self, filepath: str):
        """Load persisted index from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vector index not found at {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.chunks = data.get("chunks", [])
            texts = [f"{c.get('section', '')} {c.get('text', '')}" for c in self.chunks]
            self.embeddings = self.embedding_service.fit_transform(texts)