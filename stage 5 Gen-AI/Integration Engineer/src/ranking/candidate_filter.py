from typing import Dict, Any, List, Tuple
from ..utils.logging import get_logger

logger = get_logger("CandidateFilter")

class CandidateFilter:
    def __init__(self, min_fidelity: float = 0.80, min_faithfulness: float = 0.80):
        self.min_fidelity = min_fidelity
        self.min_faithfulness = min_faithfulness

    def is_eligible(self, record: Dict[str, Any]) -> Tuple[bool, str]:
        # 1. Structured validation
        status = record.get("status", "SUCCESS")
        if status != "SUCCESS":
            return False, f"Structured validation status is {status}"

        eval_res = record.get("evaluation", {})
        
        # 2. Fidelity
        fidelity_score = float(eval_res.get("fidelity", {}).get("fidelity_score", 1.0))
        if fidelity_score < self.min_fidelity:
            return False, f"Fidelity score {fidelity_score:.2f} < {self.min_fidelity:.2f}"

        # 3. Narrative Faithfulness
        faith_score = float(eval_res.get("narrative_faithfulness", {}).get("faithfulness_score", 1.0))
        if faith_score < self.min_faithfulness:
            return False, f"Faithfulness score {faith_score:.2f} < {self.min_faithfulness:.2f}"

        # 4. RAG quality & provenance
        rag_res = eval_res.get("rag_quality", {})
        if rag_res and not rag_res.get("provenance_valid", True):
            return False, "RAG evidence failed provenance verification"

        # 5. Reproducibility metadata
        batch_id = record.get("batch_id")
        if not batch_id:
            return False, "Missing batch_id reproducibility metadata"

        return True, "Eligible"

    def filter_candidates(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        eligible = []
        for r in records:
            ok, reason = self.is_eligible(r)
            if ok:
                eligible.append(r)
            else:
                logger.debug(f"Excluded {r.get('scenario_id')}: {reason}")
        return eligible
