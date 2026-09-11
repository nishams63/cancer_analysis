"""Metadata filter: strictly enforces approved status and domain filtering."""
from typing import List, Dict, Any


class MetadataFilter:
    def filter_chunks(self, chunks: List[Dict[str, Any]], approved_only: bool = True,
                      allowed_categories: List[str] = None) -> List[Dict[str, Any]]:
        filtered = []
        for c in chunks:
            # 1. Approval check
            if approved_only and c.get("approval_status") != "approved":
                continue
            # 2. Category check
            if allowed_categories:
                cat = c.get("evidence_category", "").lower()
                if not any(ac.lower() in cat for ac in allowed_categories):
                    continue
            filtered.append(c)
        return filtered