"""
Entity-Preserving Augmentation Controller.
Enforces strict invariants that every named entity (GENE_MUTATION, DRUG_NAME,
DOSAGE, ADVERSE_EVENT) is mathematically preserved with exact string identity
and valid, non-overlapping character offsets.
"""

from typing import List, Dict, Any, Tuple
import copy
import random
from clinical_paraphrasing import paraphrase_clinical_document


def verify_entity_spans(text: str, entities: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Validates that:
    1. 0 <= start < end <= len(text)
    2. text[start:end] == entity['text']
    3. Entities do not overlap
    """
    if not entities:
        return True, "no_entities"
        
    sorted_ents = sorted(entities, key=lambda e: e["start"])
    for i, ent in enumerate(sorted_ents):
        s, e, expected_text = ent["start"], ent["end"], ent["text"]
        if not (0 <= s < e <= len(text)):
            return False, f"invalid_bounds: span ({s}, {e}) outside text len {len(text)}"
        actual_text = text[s:e]
        if actual_text != expected_text:
            return False, f"text_mismatch: expected '{expected_text}', got '{actual_text}'"
            
        if i < len(sorted_ents) - 1:
            next_s = sorted_ents[i + 1]["start"]
            if e > next_s:
                return False, f"overlapping_spans: span ({s}, {e}) overlaps with next start {next_s}"
                
    return True, "valid"


def augment_document_preserving_entities(
    original_text: str,
    entities: List[Dict[str, Any]],
    doc_type: str,
    rng: random.Random,
    strategy: str = "composite"
) -> Tuple[str, List[Dict[str, Any]], bool, str]:
    """
    Executes entity-preserving document augmentation.
    Guarantees entity invariance before returning.
    """
    # 1. Verify original entity integrity
    orig_valid, orig_msg = verify_entity_spans(original_text, entities)
    if not orig_valid:
        return original_text, entities, False, f"original_corrupted: {orig_msg}"

    # 2. Run paraphrasing
    new_text, new_ents, success, method_desc = paraphrase_clinical_document(
        original_text, entities, doc_type, rng, strategy=strategy
    )
    if not success:
        return original_text, entities, False, method_desc

    # 3. Verify transformed entity integrity
    new_valid, new_msg = verify_entity_spans(new_text, new_ents)
    if not new_valid:
        return original_text, entities, False, f"post_transform_invalid: {new_msg}"

    # 4. Verify entity set equivalence (no entities lost or fabricated)
    orig_texts = sorted([e["text"] for e in entities])
    new_texts = sorted([e["text"] for e in new_ents])
    if orig_texts != new_texts:
        return original_text, entities, False, "entity_set_mismatch"
        
    orig_labels = sorted([e["label"] for e in entities])
    new_labels = sorted([e["label"] for e in new_ents])
    if orig_labels != new_labels:
        return original_text, entities, False, "entity_label_mismatch"

    return new_text, new_ents, True, method_desc
