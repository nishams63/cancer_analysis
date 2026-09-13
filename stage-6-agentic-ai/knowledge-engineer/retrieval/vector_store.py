"""Local Vector Store for Semantic Similarity without external API dependencies."""
from __future__ import annotations
import pickle
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from schemas.knowledge import KnowledgeItem


class LocalVectorStore:
    """High-performance, deterministic local vector store using TF-IDF subword dense n-grams.
    
    Zero-network dependency, completely isolated from external LLMs, sub-millisecond similarity.
    """

    def __init__(self, index_dir: Optional[Path | str] = None):
        if index_dir is None:
            self.index_dir = Path(__file__).resolve().parent / "index"
        else:
            self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_ids: List[str] = []
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.is_indexed: bool = False

    def build_index(self, items: List[KnowledgeItem]) -> int:
        """Fit vectorizer on all items and build normalized dense feature index."""
        if not items:
            return 0

        self.doc_ids = [item.knowledge_id for item in items]
        corpus = [item.to_searchable_text() for item in items]

        # Word (1, 2, 3)-grams with sublinear TF scaling
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            norm="l2",
            strip_accents="unicode",
            lowercase=True,
            stop_words="english",
            min_df=1,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.is_indexed = True
        self.save()
        return len(self.doc_ids)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Compute cosine similarity between query and all indexed documents."""
        if not self.is_indexed or self.vectorizer is None or self.tfidf_matrix is None:
            self.load()
            if not self.is_indexed:
                return []

        if not query or not query.strip():
            return []

        query_clean = query.strip().lower()
        query_vec = self.vectorizer.transform([query_clean])
        
        # Dense dot product of L2-normalized vectors = cosine similarity
        similarities = (query_vec * self.tfidf_matrix.T).toarray().flatten()

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results: List[Tuple[str, float]] = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0.0001:
                results.append((self.doc_ids[idx], score))

        return results

    def save(self, filepath: Optional[Path | str] = None) -> None:
        """Persist index to disk."""
        save_path = Path(filepath) if filepath else self.index_dir / "vector_store.pkl"
        payload = {
            "vectorizer": self.vectorizer,
            "doc_ids": self.doc_ids,
            "tfidf_matrix": self.tfidf_matrix,
            "is_indexed": self.is_indexed,
        }
        with open(save_path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, filepath: Optional[Path | str] = None) -> bool:
        """Load persisted index from disk."""
        load_path = Path(filepath) if filepath else self.index_dir / "vector_store.pkl"
        if not load_path.exists():
            return False
        try:
            with open(load_path, "rb") as f:
                payload = pickle.load(f)
            self.vectorizer = payload["vectorizer"]
            self.doc_ids = payload["doc_ids"]
            self.tfidf_matrix = payload["tfidf_matrix"]
            self.is_indexed = payload["is_indexed"]
            return True
        except Exception:
            return False
