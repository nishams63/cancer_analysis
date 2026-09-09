"""
High-Precision Entity Span and Annotation Propagation Module.
Tracks character offset shifts across clinical text transformations to guarantee
100% boundary integrity and exact entity text preservation.
"""

from typing import List, Dict, Any, Tuple
import copy


def propagate_entity_spans(
    original_text: str,
    entities: List[Dict[str, Any]],
    replacements: List[Tuple[int, int, str]]
) -> Tuple[str, List[Dict[str, Any]], bool]:
    """
    Apply a sorted list of non-overlapping character replacements to original_text
    while accurately projecting entity spans to their new offsets in the transformed text.
    
    replacements: List of (start_char, end_char, replacement_string)
    entities: List of dicts with 'start', 'end', 'label', 'text'
    
    Returns:
        (transformed_text, updated_entities, is_valid)
    """
    if not replacements:
        return original_text, copy.deepcopy(entities), True

    # Validate that replacements do not overlap with each other
    sorted_replacements = sorted(replacements, key=lambda r: r[0])
    for i in range(len(sorted_replacements) - 1):
        if sorted_replacements[i][1] > sorted_replacements[i + 1][0]:
            # Overlapping replacements
            return original_text, copy.deepcopy(entities), False

    # Check for collision between replacements and entity spans
    for r_start, r_end, _ in sorted_replacements:
        for ent in entities:
            e_start, e_end = ent["start"], ent["end"]
            # Any overlap between replacement and entity is strictly forbidden
            if max(r_start, e_start) < min(r_end, e_end):
                return original_text, copy.deepcopy(entities), False

    # Build transformed text and calculate delta shifts
    transformed_parts = []
    last_idx = 0
    
    # We will build a position mapping or reconstruct piecewise
    # Piecewise reconstruction:
    # We track offset delta for any point in the text
    # A cleaner approach: reconstruct text while tracking mapping from orig index to new index
    mapping = [0] * (len(original_text) + 1)
    
    new_text_chars = []
    orig_ptr = 0
    
    for r_start, r_end, r_text in sorted_replacements:
        # Copy unchanged text before replacement
        while orig_ptr < r_start:
            mapping[orig_ptr] = len(new_text_chars)
            new_text_chars.append(original_text[orig_ptr])
            orig_ptr += 1
        
        # At replacement start
        mapping[r_start] = len(new_text_chars)
        
        # Append replacement text
        new_text_chars.extend(list(r_text))
        
        # Skip original replaced characters
        while orig_ptr < r_end:
            # Replaced chars map to start of replacement or end
            orig_ptr += 1
            mapping[orig_ptr] = len(new_text_chars)

    # Append remainder
    while orig_ptr < len(original_text):
        mapping[orig_ptr] = len(new_text_chars)
        new_text_chars.append(original_text[orig_ptr])
        orig_ptr += 1
    mapping[len(original_text)] = len(new_text_chars)
    
    transformed_text = "".join(new_text_chars)

    # Project entity spans
    updated_entities = []
    for ent in entities:
        orig_start = ent["start"]
        orig_end = ent["end"]
        orig_ent_text = ent.get("text", original_text[orig_start:orig_end])
        
        new_start = mapping[orig_start]
        new_end = mapping[orig_end]
        
        # Strict validation: the text at the new span must exactly equal original entity text
        extracted_text = transformed_text[new_start:new_end]
        if extracted_text != orig_ent_text:
            return original_text, copy.deepcopy(entities), False
            
        new_ent = copy.deepcopy(ent)
        new_ent["start"] = new_start
        new_ent["end"] = new_end
        new_ent["text"] = extracted_text
        updated_entities.append(new_ent)

    return transformed_text, updated_entities, True
