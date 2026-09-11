"""Vector embedding service supporting TF-IDF cosine and API providers."""
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingService:
    def __init__(self, provider: str = "tfidf_cosine", dimension: int = 512):
        self.provider = provider
        self.dimension = dimension
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=dimension
        )
        self.is_fitted = False

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit vectorizer on corpus texts and return normalized embedding matrix."""
        if not texts:
            return np.zeros((0, self.dimension))
        matrix = self.vectorizer.fit_transform(texts).toarray()
        self.is_fitted = True
        # Normalize vectors
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform new queries using fitted vectorizer."""
        if not self.is_fitted or not texts:
            return np.zeros((len(texts), self.dimension))
        matrix = self.vectorizer.transform(texts).toarray()
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms