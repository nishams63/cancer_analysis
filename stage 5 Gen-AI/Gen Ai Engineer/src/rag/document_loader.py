"""Document and chunk loader for approved evidence corpus."""
import os
import json
from typing import List, Dict, Any


class DocumentLoader:
    def __init__(self, chunks_dir: str = None, metadata_dir: str = None):
        self.chunks_dir = chunks_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "evidence", "chunks")
        self.metadata_dir = metadata_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "evidence", "metadata")

    def load_all_chunks(self) -> List[Dict[str, Any]]:
        """Load all chunks across all chunk files."""
        chunks = []
        if not os.path.exists(self.chunks_dir):
            return chunks

        for fname in sorted(os.listdir(self.chunks_dir)):
            if fname.endswith(".json"):
                p = os.path.join(self.chunks_dir, fname)
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        chunks.extend(data)
                    elif isinstance(data, dict) and "chunks" in data:
                        chunks.extend(data["chunks"])
        return chunks

    def load_document_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load metadata for all documents."""
        metadata = {}
        if not os.path.exists(self.metadata_dir):
            return metadata

        for fname in sorted(os.listdir(self.metadata_dir)):
            if fname.endswith(".json"):
                p = os.path.join(self.metadata_dir, fname)
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    doc_id = data.get("document_id", fname.replace("_metadata.json", ""))
                    metadata[doc_id] = data
        return metadata