"""
Clinical Paraphrasing Orchestrator.
Composes safe terminology substitutions, clause reordering, and context framing
to generate natural clinical diversity while maintaining mathematical span integrity.
"""

from typing import List, Dict, Any, Tuple
import random
from terminology_augmentation import apply_terminology_augmentation
from sentence_augmentation import apply_sentence_restructuring
from context_augmentation import apply_context_framing_augmentation


def paraphrase_clinical_document(
    text: str,
    entities: List[Dict[str, Any]],
    doc_type: str,
    rng: random.Random,
    strategy: str = "composite"
) -> Tuple[str, List[Dict[str, Any]], bool, str]:
    """
    Apply clinically safe paraphrasing using designated or composite strategy.
    Strategies:
    - 'terminology': Carrier phrase synonyms only
    - 'restructure': Clause/sentence permuting only
    - 'context': Context framing variations only
    - 'composite': Combines terminology + restructuring
    """
    if strategy == "terminology":
        return apply_terminology_augmentation(text, entities, rng)
    elif strategy == "restructure":
        return apply_sentence_restructuring(text, entities, rng)
    elif strategy == "context":
        return apply_context_framing_augmentation(text, entities, rng)
    elif strategy == "composite":
        # Step 1: Attempt terminology
        cur_text, cur_ents, s1, desc1 = apply_terminology_augmentation(text, entities, rng)
        # Step 2: Attempt restructuring on the result
        cur_text, cur_ents, s2, desc2 = apply_sentence_restructuring(cur_text, cur_ents, rng)
        # Step 3: If neither worked, try context
        if not s1 and not s2:
            return apply_context_framing_augmentation(text, entities, rng)
            
        desc = []
        if s1: desc.append(desc1)
        if s2: desc.append(desc2)
        return cur_text, cur_ents, True, "+".join(desc)
    else:
        raise ValueError(f"Unknown paraphrasing strategy: {strategy}")
