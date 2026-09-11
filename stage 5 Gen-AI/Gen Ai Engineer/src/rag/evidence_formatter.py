"""Evidence Formatter: formats retrieved chunks with citations for LLM prompt context."""
from typing import List, Dict, Any


class EvidenceFormatter:
    def format_evidence_block(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        if not retrieved_chunks:
            return "No supporting clinical evidence retrieved."

        lines = ["=== APPROVED CLINICAL EVIDENCE CONTEXT ==="]
        for idx, c in enumerate(retrieved_chunks, 1):
            cid = c.get("chunk_id", "N/A")
            src = c.get("source", "Approved_Source")
            ver = c.get("version", "1.0")
            cat = c.get("evidence_category", "guideline")
            sec = c.get("section", "Clinical Context")
            score = c.get("retrieval_score", 0.0)
            text = c.get("text", "").strip()

            lines.append(f"\n[CITATION {idx}] (CHUNK_ID: {cid} | Source: {src} v{ver} | Category: {cat} | Section: {sec} | Score: {score})")
            lines.append(text)

        lines.append("\n=== END OF EVIDENCE CONTEXT ===")
        return "\n".join(lines)

    format_context = format_evidence_block