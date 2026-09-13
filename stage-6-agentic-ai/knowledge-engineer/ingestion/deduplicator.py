"""Exact and Near Duplicate Detection Engine."""
import hashlib
from typing import List, Dict, Set, Tuple
from schemas.knowledge import KnowledgeItem


class KnowledgeDeduplicator:
    """Prevents duplicate pollution and logs semantic duplicates."""

    def __init__(self):
        self.seen_ids: Set[str] = set()
        self.seen_hashes: Dict[str, str] = {}  # hash -> knowledge_id

    @staticmethod
    def compute_hash(item: KnowledgeItem) -> str:
        content = (
            f"{item.title.lower().strip()}|{item.category.value}|{item.subcategory.lower().strip()}|"
            f"{item.description.lower().strip()}"
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def check_duplicate(self, item: KnowledgeItem) -> Tuple[bool, str]:
        """Check if item is exact duplicate by ID or normalized content hash."""
        k_id = item.knowledge_id.upper().strip()
        c_hash = self.compute_hash(item)

        if k_id in self.seen_ids:
            return True, f"Duplicate knowledge_id '{k_id}'"

        if c_hash in self.seen_hashes:
            orig_id = self.seen_hashes[c_hash]
            return True, f"Identical content match with existing item '{orig_id}'"

        self.seen_ids.add(k_id)
        self.seen_hashes[c_hash] = k_id
        return False, "ok"
