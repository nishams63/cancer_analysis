"""Modular candidate chunk reranker."""
from typing import List, Dict, Any


class Reranker:
    """Optional candidate reranker weighting keyword matches."""
    def rerank(self, query_spec: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        req_kws = [kw.lower() for kw in query_spec.get("required_keywords", [])]
        for c in candidates:
            text = c.get("text", "").lower()
            bonus = sum(0.15 for kw in req_kws if kw in text)
            c["reranked_score"] = round(c.get("retrieval_score", 0.0) + bonus, 4)

        return sorted(candidates, key=lambda x: x.get("reranked_score", 0.0), reverse=True)