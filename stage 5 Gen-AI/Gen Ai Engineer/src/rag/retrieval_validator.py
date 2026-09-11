"""RAG Retrieval Validator and Conflict Detector."""
from typing import List, Dict, Any, Tuple


class RetrievalValidator:
    def validate_retrieval(self, query_spec: Dict[str, Any], retrieved_chunks: List[Dict[str, Any]],
                           patient: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        violations = []
        conflicts = []

        # 1. Non-empty check
        if not retrieved_chunks:
            violations.append("Retrieval failed: Zero evidence chunks retrieved.")
            return False, violations, {"conflicts": conflicts}

        # 2. Approved status check
        for c in retrieved_chunks:
            if c.get("approval_status") != "approved":
                violations.append(f"Unapproved evidence chunk retrieved: {c.get('chunk_id')}")

        # 3. Required keywords / categories check
        req_kws = query_spec.get("required_keywords", [])
        matched_kws = set()
        for c in retrieved_chunks:
            text = c.get("text", "").lower()
            for kw in req_kws:
                if kw.lower() in text:
                    matched_kws.add(kw)

        missing_kws = set(req_kws) - matched_kws
        # If mandatory keywords missing, note coverage
        coverage_pct = round((len(matched_kws) / len(req_kws)) * 100, 1) if req_kws else 100.0

        # 4. Conflict detection against patient ground truth
        # e.g., if chunk asserts "monotherapy only" while patient has dual resistance
        for c in retrieved_chunks:
            text = c.get("text", "").lower()
            if "monotherapy chemotherapy only" in text and len(patient.get("mutations", [])) > 1:
                conflicts.append({
                    "chunk_id": c.get("chunk_id"),
                    "conflict_type": "contradicts_dual_resistance",
                    "chunk_context": "Recommends monotherapy only"
                })

        is_valid = (len(violations) == 0) and (len(retrieved_chunks) >= query_spec.get("min_documents", 1))

        details = {
            "retrieved_count": len(retrieved_chunks),
            "matched_keywords": list(matched_kws),
            "missing_keywords": list(missing_kws),
            "coverage_pct": coverage_pct,
            "conflicts": conflicts
        }
        return is_valid, violations, details