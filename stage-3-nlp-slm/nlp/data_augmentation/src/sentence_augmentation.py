"""
Controlled Sentence and Clause Restructuring Augmentation Module.
Permutes independent clinical measurements (vitals metrics, laboratory values,
independent pre-infusion checks) while preserving clinical causality and entity spans.
"""

from typing import List, Dict, Any, Tuple, Optional
import random
import re
from annotation_propagation import propagate_entity_spans


def permute_vitals_clauses(
    text: str,
    entities: List[Dict[str, Any]],
    rng: random.Random
) -> Tuple[str, List[Dict[str, Any]], bool, str]:
    """
    Reorders independent vital sign components within vitals lines.
    E.g., "Blood pressure X, Heart rate Y, SpO2 Z" -> "Heart rate Y, Blood pressure X, SpO2 Z"
    """
    # Pattern matching consultation vitals line
    vitals_pattern = re.compile(
        r"(Vitals:\s+)(Blood pressure\s+[^,]+),\s+(Heart rate\s+[^,]+),\s+(SpO2\s+[^.\n]+)([.|\n])"
    )
    match = vitals_pattern.search(text)
    if match:
        v_start, v_end = match.start(), match.end()
        # Entity guard
        if any(max(v_start, e["start"]) < min(v_end, e["end"]) for e in entities):
            return text, entities, False, "entity_collision"
            
        prefix, bp, hr, spo2, punct = match.groups()
        items = [bp, hr, spo2]
        rng.shuffle(items)
        # Ensure it's not identical to original
        if items == [bp, hr, spo2]:
            items = [hr, bp, spo2]
            
        new_vitals = f"{prefix}{items[0]}, {items[1]}, {items[2]}{punct}"
        new_text, new_ents, valid = propagate_entity_spans(text, entities, [(v_start, v_end, new_vitals)])
        if valid:
            return new_text, new_ents, True, "vitals_permutation_consult"

    # Pattern matching nurse intake vitals line
    nurse_vitals = re.compile(
        r"(Vital Signs:\s+)(BP\s+[^,]+),\s+(Pulse\s+[^,]+),\s+(Resp Rate\s+[^,]+),\s+(SpO2\s+[^.\n]+)([.|\n])"
    )
    match2 = nurse_vitals.search(text)
    if match2:
        v_start, v_end = match2.start(), match2.end()
        if any(max(v_start, e["start"]) < min(v_end, e["end"]) for e in entities):
            return text, entities, False, "entity_collision"
            
        prefix, bp, pulse, resp, spo2, punct = match2.groups()
        items = [bp, pulse, resp, spo2]
        rng.shuffle(items)
        if items == [bp, pulse, resp, spo2]:
            items = [pulse, bp, resp, spo2]
            
        new_vitals = f"{prefix}{', '.join(items)}{punct}"
        new_text, new_ents, valid = propagate_entity_spans(text, entities, [(v_start, v_end, new_vitals)])
        if valid:
            return new_text, new_ents, True, "vitals_permutation_nurse"

    return text, entities, False, "no_vitals_matched"


def permute_lab_chemistry_clauses(
    text: str,
    entities: List[Dict[str, Any]],
    rng: random.Random
) -> Tuple[str, List[Dict[str, Any]], bool, str]:
    """
    Reorders independent chemistry labs in laboratory review lines:
    E.g. Serum creatinine, Liver function enzymes, Hemoglobin.
    """
    lab_pattern = re.compile(
        r"(Serum Chemistries:\s+)(Serum creatinine\s+[^,]+),\s+(Liver function enzymes\s+[^,]+),\s+(Hemoglobin\s+[^.\n]+)([.|\n])"
    )
    match = lab_pattern.search(text)
    if not match:
        return text, entities, False, "no_labs_matched"
        
    l_start, l_end = match.start(), match.end()
    # If any entity falls inside this line (e.g. ADVERSE_EVENT on lab), strictly abort
    if any(max(l_start, e["start"]) < min(l_end, e["end"]) for e in entities):
        return text, entities, False, "entity_collision"
        
    prefix, cr, lft, hgb, punct = match.groups()
    items = [cr, lft, hgb]
    rng.shuffle(items)
    if items == [cr, lft, hgb]:
        items = [lft, cr, hgb]
        
    new_labs = f"{prefix}{items[0]}, {items[1]}, {items[2]}{punct}"
    new_text, new_ents, valid = propagate_entity_spans(text, entities, [(l_start, l_end, new_labs)])
    if valid:
        return new_text, new_ents, True, "lab_chemistry_permutation"
        
    return text, entities, False, "lab_propagation_failed"


def apply_sentence_restructuring(
    text: str,
    entities: List[Dict[str, Any]],
    rng: random.Random
) -> Tuple[str, List[Dict[str, Any]], bool, str]:
    """
    Attempts clinically sound sentence/clause restructurings:
    1. Vitals permutation
    2. Lab chemistry permutation
    """
    methods = [permute_vitals_clauses, permute_lab_chemistry_clauses]
    rng.shuffle(methods)
    
    for method in methods:
        new_text, new_ents, success, desc = method(text, entities, rng)
        if success:
            return new_text, new_ents, True, desc
            
    return text, entities, False, "no_restructuring_applicable"
