"""Schema Validator and Quality Scorer for Knowledge Items."""
from typing import List, Tuple, Dict, Any, Optional
from pydantic import ValidationError
from schemas.knowledge import KnowledgeItem


class KnowledgeValidator:
    """Validates dictionary records against KnowledgeItem Pydantic specification."""

    @staticmethod
    def validate_item(raw_dict: Dict[str, Any]) -> Tuple[Optional[KnowledgeItem], Optional[str]]:
        """Validate single dict. Returns (item, error_str)."""
        try:
            item = KnowledgeItem(**raw_dict)
            if item.quality_score is None:
                item.quality_score = item.compute_quality_score()
            return item, None
        except ValidationError as ve:
            errors = []
            for err in ve.errors():
                loc = ".".join(str(p) for p in err.get("loc", []))
                msg = err.get("msg", "Invalid field")
                errors.append(f"{loc}: {msg}")
            return None, "; ".join(errors)
        except Exception as ex:
            return None, str(ex)

    def validate_batch(
        self, raw_items: List[Dict[str, Any]]
    ) -> Tuple[List[KnowledgeItem], List[Tuple[Dict[str, Any], str]]]:
        """Validate a collection of raw item dicts."""
        valid = []
        rejected = []
        for raw in raw_items:
            item, err = self.validate_item(raw)
            if item is not None:
                valid.append(item)
            else:
                rejected.append((raw, err or "Unknown validation error"))
        return valid, rejected
