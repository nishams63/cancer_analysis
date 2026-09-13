"""Hybrid Ranker combining Semantic Similarity, Keyword Matches, and Metadata Boosts."""
from __future__ import annotations
import re
from typing import List, Dict, Set
from schemas.knowledge import KnowledgeItem, SearchResult


class HybridRanker:
    """Re-ranks candidates using weighted semantic similarity and exact term matching."""

    def __init__(self, semantic_weight: float = 0.65, keyword_weight: float = 0.35):
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

    @staticmethod
    def _extract_keywords(text: str) -> Set[str]:
        tokens = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text.lower())
        stopwords = {
            "the", "and", "for", "with", "this", "that", "how", "what", "should",
            "can", "use", "when", "data", "analysis", "are", "from", "which"
        }
        return {t for t in tokens if t not in stopwords}

    def compute_keyword_score(self, query: str, item: KnowledgeItem) -> float:
        """Compute normalized keyword overlap score."""
        query_words = self._extract_keywords(query)
        if not query_words:
            return 0.0

        title_words = self._extract_keywords(item.title)
        desc_words = self._extract_keywords(item.description)
        methods_words = self._extract_keywords(" ".join(item.recommended_methods))
        rules_words = self._extract_keywords(" ".join(item.selection_rules))

        score = 0.0
        for word in query_words:
            if word in title_words:
                score += 0.40
            if word in methods_words:
                score += 0.30
            if word in desc_words:
                score += 0.20
            if word in rules_words:
                score += 0.20

        # Normalize relative to query length
        max_possible = len(query_words) * 0.70
        return round(min(score / max(max_possible, 1.0), 1.0), 4)

    def rank(
        self,
        query: str,
        items: List[KnowledgeItem],
        semantic_scores: Dict[str, float],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """Combine semantic and keyword scores, sort, and return top_k SearchResult objects."""
        scored_candidates = []
        for item in items:
            sem_score = semantic_scores.get(item.knowledge_id, 0.0)
            kw_score = self.compute_keyword_score(query, item)

            combined_score = round(
                (self.semantic_weight * sem_score) + (self.keyword_weight * kw_score), 4
            )

            # Bonus for exact phrase match in title or category
            if query.lower().strip() in item.title.lower():
                combined_score = round(min(combined_score + 0.15, 1.0), 4)

            if combined_score >= min_score:
                scored_candidates.append((combined_score, sem_score, kw_score, item))

        # Sort by score descending, then by quality_score descending
        scored_candidates.sort(
            key=lambda t: (t[0], t[3].quality_score or 0.0), reverse=True
        )

        # Build SearchResult with 1-based ranks
        final_results: List[SearchResult] = []
        for i, (comb_score, sem_s, kw_s, item) in enumerate(scored_candidates[:top_k], start=1):
            final_results.append(
                SearchResult(
                    item=item,
                    score=comb_score,
                    semantic_score=round(sem_s, 4),
                    keyword_score=round(kw_s, 4),
                    rank=i,
                )
            )

        return final_results
