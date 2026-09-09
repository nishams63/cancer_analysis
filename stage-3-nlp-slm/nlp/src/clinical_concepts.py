"""
Clinical Concept Extraction Module for Stage 3 NLP.
Extracts antineoplastic drugs, genomic driver mutations, dosages, and adverse events
aligned with the 4-label taxonomy: GENE_MUTATION, DRUG_NAME, DOSAGE, ADVERSE_EVENT.
"""

import re
from typing import List, Dict, Any, Tuple
from negation_detection import resolve_concept_polarity

# Standardized Concept Dictionaries Grounded in Data Engineering Vocabulary
DRUG_PATTERNS = [
    r"\b(?:cisplatin|carboplatin|oxaliplatin|paclitaxel|docetaxel)\b",
    r"\b(?:pembrolizumab|nivolumab|atezolizumab|durvalumab)\b",
    r"\b(?:osimertinib|erlotinib|gefitinib|alectinib|crizotinib)\b",
    r"\b(?:fluorouracil|5-fu|capecitabine|gemcitabine|doxorubicin)\b",
    r"\b(?:trastuzumab|pertuzumab|tamoxifen|letrozole|enzalutamide)\b",
    r"\b(?:abiraterone|irinotecan|etoposide)\b"
]

MUTATION_PATTERNS = [
    r"\b(?:egfr|kras|tp53|braf|alk|pik3ca|her2|erbb2|brca1|brca2|ros1|met|ret)\b",
    r"\b(?:t790m|g12d|g12c|v600e|exon\s*19\s*del|l858r)\b",
    r"\bNone\/Unknown\b"
]

DOSAGE_PATTERNS = [
    r"\b\d+(?:\.\d+)?\s*(?:mg\/m2|mg\/dL|mg|mcg|g\/dL|mmHg|%)\b"
]

ADVERSE_EVENT_PATTERNS = [
    r"\b(?:no\s+acute\s+adverse\s+toxicities|manageable\s+mild\s+fatigue)\b",
    r"\b(?:neutropenia|thrombocytopenia|anemia|leukopenia)\b",
    r"\b(?:dyspnea|shortness\s+of\s+breath|cough|pneumonitis)\b",
    r"\b(?:fatigue|asthenia|malaise|lethargy)\b",
    r"\b(?:nausea|vomiting|emesis|diarrhea|constipation)\b",
    r"\b(?:rash|pruritus|dermatitis|alopecia)\b",
    r"\b(?:neuropathy|paresthesia|numbness|tingling)\b",
    r"\b(?:elevated\s+transaminases|hepatotoxicity|hyperbilirubinemia)\b",
    r"\b(?:nephrotoxicity|acute\s+kidney\s+injury|elevated\s+creatinine)\b",
    r"\b(?:cardiotoxicity|arrhythmia|heart\s+failure)\b",
    r"\b(?:breast\s+cancer|nsclc|colorectal\s+cancer|prostate\s+cancer|pancreatic\s+cancer)\b"
]


def extract_clinical_concepts(text: str, assign_polarity: bool = True) -> List[Dict[str, Any]]:
    """
    Scan clinical text and extract all concept mentions matching the 4-label taxonomy:
    GENE_MUTATION, DRUG_NAME, DOSAGE, ADVERSE_EVENT.
    Returns: list of dicts with start, end, label, text, and polarity.
    """
    if not text or not isinstance(text, str):
        return []

    entities = []

    # 1. Extract Drugs
    for pat in DRUG_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            entities.append({
                "start": m.start(),
                "end": m.end(),
                "label": "DRUG_NAME",
                "text": text[m.start():m.end()]
            })

    # 2. Extract Mutations
    for pat in MUTATION_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            entities.append({
                "start": m.start(),
                "end": m.end(),
                "label": "GENE_MUTATION",
                "text": text[m.start():m.end()]
            })

    # 3. Extract Dosages
    for pat in DOSAGE_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            entities.append({
                "start": m.start(),
                "end": m.end(),
                "label": "DOSAGE",
                "text": text[m.start():m.end()]
            })

    # 4. Extract Adverse Events & Symptoms
    for pat in ADVERSE_EVENT_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            entities.append({
                "start": m.start(),
                "end": m.end(),
                "label": "ADVERSE_EVENT",
                "text": text[m.start():m.end()]
            })

    # De-duplicate overlapping spans (prefer longer span)
    entities = sorted(entities, key=lambda e: (e["start"], -(e["end"] - e["start"])))
    filtered = []
    last_end = -1
    for ent in entities:
        if ent["start"] >= last_end:
            filtered.append(ent)
            last_end = ent["end"]

    # Assign clinical polarity
    if assign_polarity:
        for ent in filtered:
            polarity = resolve_concept_polarity(text, ent["start"], ent["end"])
            ent["polarity"] = polarity

    return filtered


def evaluate_entity_spans(predicted_entities: List[Dict[str, Any]],
                          ground_truth_entities: List[Dict[str, Any]],
                          match_type: str = "exact") -> Dict[str, float]:
    """
    Evaluate span-level Precision, Recall, and F1 score against ground truth.
    Supports 'exact' match (same start, end, label) or 'relaxed' match (span overlap + same label).
    """
    if not ground_truth_entities and not predicted_entities:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not ground_truth_entities:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    if not predicted_entities:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}

    true_positives = 0
    matched_gt = set()

    for p in predicted_entities:
        for i, gt in enumerate(ground_truth_entities):
            if i in matched_gt:
                continue
            if p["label"] == gt["label"]:
                if match_type == "exact":
                    if p["start"] == gt["start"] and p["end"] == gt["end"]:
                        true_positives += 1
                        matched_gt.add(i)
                        break
                elif match_type == "relaxed":
                    # Check span overlap
                    if max(p["start"], gt["start"]) < min(p["end"], gt["end"]):
                        true_positives += 1
                        matched_gt.add(i)
                        break

    precision = true_positives / len(predicted_entities) if predicted_entities else 0.0
    recall = true_positives / len(ground_truth_entities) if ground_truth_entities else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": true_positives,
        "predicted_count": len(predicted_entities),
        "ground_truth_count": len(ground_truth_entities)
    }
