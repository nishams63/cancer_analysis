"""RAG Retrieval Quality & Provenance Evaluator."""
import os
import json
from pathlib import Path
from typing import Dict, Any, List


class RAGQualityEvaluator:
    """Evaluates Level 2: RAG Retrieval Relevance, Concept Coverage, Provenance, and Conflict."""

    def __init__(self, chunks_dir: str | Path = None):
        if not chunks_dir:
            # Default to Data Engineer evidence chunks
            chunks_dir = Path(__file__).resolve().parent.parent.parent.parent / "Data Engineer" / "data" / "evidence" / "chunks"
        self.chunks_dir = Path(chunks_dir)
        self.chunk_store = {}
        self._load_chunk_store()

    def _load_chunk_store(self):
        if self.chunks_dir.exists():
            for c_file in self.chunks_dir.glob("*_chunks.json"):
                try:
                    with open(c_file, "r", encoding="utf-8") as f:
                        chunks = json.load(f)
                        for c in chunks:
                            cid = c.get("chunk_id")
                            if cid:
                                self.chunk_store[cid] = c
                except Exception:
                    pass

    def evaluate_retrieval(self, retrieval_entry: Dict[str, Any], scenario_def: Dict[str, Any] = None) -> Dict[str, Any]:
        sid = retrieval_entry.get("scenario_id", "UNKNOWN")
        
        # Support both 'retrieved_chunks' (list of IDs) and 'results' (list of chunk dicts)
        raw_chunks = retrieval_entry.get("results", [])
        chunk_ids = retrieval_entry.get("retrieved_chunks", [])
        scores = retrieval_entry.get("scores", [])

        resolved_chunks = []
        if raw_chunks:
            resolved_chunks = raw_chunks
        elif chunk_ids:
            for idx, cid in enumerate(chunk_ids):
                if cid in self.chunk_store:
                    c_dict = dict(self.chunk_store[cid])
                    if idx < len(scores):
                        c_dict["score"] = scores[idx]
                    resolved_chunks.append(c_dict)
                else:
                    resolved_chunks.append({
                        "chunk_id": cid,
                        "document_id": cid.split("-")[1] if "-" in cid else "UNKNOWN",
                        "approval_status": "approved",
                        "score": scores[idx] if idx < len(scores) else 0.35,
                        "text": ""
                    })

        # 1. Provenance validation
        provenance_failures = []
        for r in resolved_chunks:
            cid = r.get("chunk_id")
            doc_id = r.get("document_id")
            status = r.get("approval_status", "approved")
            if not cid or not doc_id:
                provenance_failures.append(f"Missing ID in chunk {cid}")
            if status != "approved":
                provenance_failures.append(f"Unapproved source in chunk {cid}")

        has_provenance = (len(provenance_failures) == 0 and len(resolved_chunks) > 0)

        # 2. Concept coverage
        required_concepts = []
        if scenario_def:
            entities = scenario_def.get("required_entities", [])
            for e in entities:
                required_concepts.extend([v.lower() for v in e.get("values", [])])
        if not required_concepts:
            query = retrieval_entry.get("query", "")
            required_concepts = [w.lower() for w in query.split() if len(w) > 3]

        covered_concepts = set()
        for r in resolved_chunks:
            txt = (r.get("text", "") + " " + r.get("section", "")).lower()
            for c in required_concepts:
                if c in txt or any(w in txt for w in c.split()):
                    covered_concepts.add(c)

        coverage_ratio = len(covered_concepts) / max(1, len(required_concepts))
        
        # 3. Mean score
        c_scores = [r.get("score", 0.0) for r in resolved_chunks]
        mean_score = sum(c_scores) / max(1, len(c_scores)) if c_scores else 0.0

        # RAG failure classification
        failure_codes = []
        if coverage_ratio < 0.40:
            failure_codes.append("R01")
        if mean_score < 0.15:
            failure_codes.append("R02")
        if any("Unapproved" in f for f in provenance_failures):
            failure_codes.append("R04")
        if any("Missing ID" in f for f in provenance_failures):
            failure_codes.append("R05")

        overall_status = "PASS" if (has_provenance and coverage_ratio >= 0.40) else "FAIL"

        return {
            "scenario_id": sid,
            "status": overall_status,
            "total_chunks_retrieved": len(resolved_chunks),
            "mean_retrieval_score": round(mean_score, 4),
            "concept_coverage_ratio": round(coverage_ratio, 4),
            "provenance_valid": has_provenance,
            "failure_codes": failure_codes,
            "covered_concepts": list(covered_concepts),
            "missing_concepts": list(set(required_concepts) - covered_concepts)
        }
