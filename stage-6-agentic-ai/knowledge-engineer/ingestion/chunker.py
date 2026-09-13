"""Knowledge Chunker for Splitting Composite Files into Atomic Units."""
from typing import List, Dict, Any


class KnowledgeChunker:
    """Disaggregates files containing multiple knowledge entries or composite guide sections."""

    @staticmethod
    def chunk(data: Any) -> List[Dict[str, Any]]:
        """Flatten either single dict or array of dicts into list of discrete knowledge units."""
        if isinstance(data, list):
            items = []
            for entry in data:
                if isinstance(entry, dict):
                    items.append(entry)
            return items
        elif isinstance(data, dict):
            # Check if file wraps items in a 'knowledge_items' or 'items' list
            for key in ["knowledge_items", "items", "rules"]:
                if key in data and isinstance(data[key], list):
                    return [i for i in data[key] if isinstance(i, dict)]
            return [data]
        return []
