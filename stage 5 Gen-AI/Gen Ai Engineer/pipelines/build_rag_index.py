"""RAG Index Builder: embeds approved evidence chunks and persists vector index."""
import sys
import os
sys.path.insert(0, os.path.abspath("."))

from src.rag import DocumentLoader, VectorStore, EmbeddingService
from src.provenance import RAGLineageTracker


def build_rag_index():
    print("=" * 70)
    print("BUILDING RAG EVIDENCE VECTOR INDEX")
    print("=" * 70)

    index_out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "retrieval", "indexes", "evidence_vector_index.json")
    manifest_out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "manifests", "rag_manifest.json")

    loader = DocumentLoader()
    chunks = loader.load_all_chunks()
    print(f"Loaded {len(chunks)} evidence chunks from data/evidence/chunks/")

    # Filter approved only
    approved_chunks = [c for c in chunks if c.get("approval_status") == "approved"]
    print(f"Approved chunks for indexing: {len(approved_chunks)}/{len(chunks)}")

    store = VectorStore()
    n_indexed = store.build_index(approved_chunks)
    store.save(index_out)
    print(f"Persisted vector index to {index_out} ({n_indexed} chunks indexed)")

    lineage = RAGLineageTracker()
    lineage.export_rag_manifest(manifest_out, index_out, n_indexed, [])
    print(f"Generated RAG manifest at {manifest_out}")
    print("=" * 70)
    return index_out


if __name__ == "__main__":
    build_rag_index()