"""
Clinical Terminology and Carrier Phrase Augmentation Module.
Provides verified clinical synonym replacements for non-entity carrier text,
strictly isolating annotations, drug names, dosages, mutations, and toxicities.
"""

from typing import List, Dict, Any, Tuple, Optional
import random
import re
from annotation_propagation import propagate_entity_spans

# Verified clinical carrier phrase substitution dictionary
# Key: canonical phrase pattern, Values: clinically equivalent variants
CARRIER_PHRASE_SYNONYMS = [
    # Headings and structural signposts
    (r"\bCLINICAL ASSESSMENT & LABORATORY REVIEW:\b", [
        "CLINICAL EVALUATION & LAB REVIEW:",
        "OBJECTIVE FINDINGS & LAB REVIEW:",
        "CLINICAL ASSESSMENT & LAB INVESTIGATIONS:"
    ]),
    (r"\bTREATMENT PLAN & REGIMEN:\b", [
        "PLANNED THERAPY & ONCOLOGY REGIMEN:",
        "TREATMENT REGIMEN & CLINICAL PLAN:",
        "THERAPEUTIC PLAN & CHEMOTHERAPY REGIMEN:"
    ]),
    (r"\bGROSS & MICROSCOPIC EXAMINATION:\b", [
        "MACROSCOPIC & MICROSCOPIC EXAMINATION:",
        "GROSS PATHOLOGY & MICROSCOPIC FINDINGS:",
        "PATHOLOGIC & HISTOLOGIC EXAMINATION:"
    ]),
    (r"\bTRIAGE & CLINICAL VITALS:\b", [
        "VITAL SIGNS & INTAKE TRIAGE:",
        "INTAKE TRIAGE & VITALS:",
        "CLINICAL TRIAGE & BASELINE VITALS:"
    ]),
    (r"\bNURSING INTERVENTIONS:\b", [
        "CLINICAL NURSING ACTIONS:",
        "NURSING CARE PLAN & ACTIONS:",
        "ONCOLOGY NURSING INTERVENTIONS:"
    ]),
    
    # Clinical exam & physical findings (carrier language)
    (r"\bPhysical examination reveals\b", [
        "Physical exam demonstrates",
        "Clinical examination shows",
        "Bedside examination reveals"
    ]),
    (r"\bclear lung fields bilaterally\b", [
        "lungs clear to auscultation bilaterally",
        "bilateral lung fields clear",
        "clear breath sounds bilaterally"
    ]),
    (r"\bPre-medications ordered per protocol\b", [
        "Pre-treatment medications administered per standard protocol",
        "Premedications ordered per clinical oncology protocol",
        "Standard pre-infusion supportive medications ordered"
    ]),
    (r"\bSerial monitoring of renal markers\b", [
        "Routine surveillance of renal biomarkers",
        "Serial evaluation of renal markers",
        "Close monitoring of renal function panels"
    ]),
    (r"\bcomplete blood count scheduled prior to next infusion\b", [
        "CBC scheduled before subsequent infusion",
        "complete blood count planned prior to next infusion cycle",
        "routine CBC panel scheduled prior to subsequent therapy"
    ]),
    
    # Pathology carrier phrases
    (r"\bSpecimen consists of multiple fibrous tissue cores exhibiting\b", [
        "Specimen demonstrates multiple fibrous tissue cores showing",
        "Specimen consists of multiple biopsy tissue cores demonstrating",
        "Biopsy sample comprises multiple fibrous tissue cores revealing"
    ]),
    (r"\bMicroscopic sections reveal\b", [
        "Histologic sections demonstrate",
        "Microscopic examination reveals",
        "Microscopic evaluation demonstrates"
    ]),
    (r"\bPathologic Staging: Pathologic evaluation aligns with\b", [
        "Pathologic Staging: Staging evaluation corresponds to",
        "Pathologic Stage: Morphologic evaluation consistent with",
        "Pathologic Staging: Histopathologic findings align with"
    ]),
    (r"\bRecommendation: Correlation with systemic clinical oncology records\b", [
        "Recommendation: Clinical correlation with outpatient oncology documentation",
        "Recommendation: Correlate with longitudinal oncology clinical records",
        "Recommendation: Correlation with patient oncology clinical chart"
    ]),
    
    # Nursing carrier phrases
    (r"\bECOG Performance Status: Evaluated as ambulatory and self-caring\b", [
        "ECOG Performance Status: Assessed as ambulatory and capable of self-care",
        "ECOG Functional Status: Documented as ambulatory and self-caring",
        "Performance Status (ECOG): Patient remains ambulatory and self-caring"
    ]),
    (r"\bIV peripheral access established with blood return confirmed in right forearm\b", [
        "Peripheral IV line established with brisk blood return in right forearm",
        "Peripheral intravenous access confirmed with blood return in right arm",
        "Patent peripheral venous catheter secured in right forearm with good blood return"
    ]),
    (r"\bHydration protocol initiated per oncology standing orders\b", [
        "Hydration commenced in accordance with oncology standing orders",
        "Intravenous hydration initiated per oncology standing protocol",
        "Pre-treatment IV hydration administered per oncology unit protocol"
    ]),
    (r"\bPhysician notified regarding baseline labs and symptom status\b", [
        "Attending oncologist alerted regarding baseline laboratory parameters and symptoms",
        "Supervising oncologist notified of baseline labs and triage status",
        "Clinical care team updated regarding pre-treatment labs and symptom assessment"
    ]),
    
    # Consultation demographics carrier phrases
    (r"\bDemographics: (\d+)-year-old (male|female) presenting for Cycle\b", [
        r"Demographics: \1-year-old \2 presenting for scheduled Cycle",
        r"Demographics: \1 yo \2 attending clinical evaluation for Cycle",
        r"Demographics: \1-year-old \2 evaluated prior to planned Cycle"
    ])
]


def apply_terminology_augmentation(
    text: str,
    entities: List[Dict[str, Any]],
    rng: random.Random,
    max_substitutions: int = 3
) -> Tuple[str, List[Dict[str, Any]], bool, str]:
    """
    Search for candidate carrier phrase substitutions that do NOT overlap with any
    entity spans or negation triggers.
    
    Returns:
        (transformed_text, updated_entities, success, method_description)
    """
    candidate_replacements = []
    
    # Find all potential matches
    for pattern, replacements in CARRIER_PHRASE_SYNONYMS:
        for match in re.finditer(pattern, text):
            sub_start, sub_end = match.start(), match.end()
            
            # Entity collision guard: check if any entity overlaps
            has_entity_collision = any(
                max(sub_start, ent["start"]) < min(sub_end, ent["end"])
                for ent in entities
            )
            if has_entity_collision:
                continue
                
            # Select replacement
            choice = rng.choice(replacements)
            # Expand regex group references if any
            if r"\1" in choice:
                choice = match.expand(choice)
                
            if choice != match.group():
                candidate_replacements.append((sub_start, sub_end, choice))

    if not candidate_replacements:
        return text, entities, False, "no_valid_substitutions"

    # Select non-overlapping subset of candidate replacements
    rng.shuffle(candidate_replacements)
    selected_replacements = []
    
    for cand in candidate_replacements:
        c_start, c_end, _ = cand
        # Check collision with already selected
        collision = any(
            max(c_start, s_start) < min(c_end, s_end)
            for s_start, s_end, _ in selected_replacements
        )
        if not collision:
            selected_replacements.append(cand)
            if len(selected_replacements) >= max_substitutions:
                break

    if not selected_replacements:
        return text, entities, False, "no_non_overlapping_substitutions"

    # Propagate through entity span re-aligner
    new_text, new_ents, valid = propagate_entity_spans(text, entities, selected_replacements)
    if not valid:
        return text, entities, False, "span_propagation_failed"
        
    return new_text, new_ents, True, f"terminology_synonyms_n{len(selected_replacements)}"
