"""Knowledge Item Parser and Normalizer."""
from typing import Dict, Any, List


class KnowledgeParser:
    """Normalizes raw dictionary representations of knowledge items."""

    @staticmethod
    def normalize_raw_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {}
        for k, v in raw.items():
            key = k.strip().lower()
            if isinstance(v, str):
                normalized[key] = v.strip()
            elif isinstance(v, list):
                normalized[key] = [item.strip() if isinstance(item, str) else item for item in v]
            else:
                normalized[key] = v

        # Default fallback values for optional metadata
        if "source" not in normalized or not normalized["source"]:
            normalized["source"] = "AADA Internal Methodology"
        if "version" not in normalized or not normalized["version"]:
            normalized["version"] = "1.0"
        if "status" not in normalized or not normalized["status"]:
            normalized["status"] = "active"
        if "effective_date" not in normalized or not normalized["effective_date"]:
            normalized["effective_date"] = "2026-01-01"

        return normalized
