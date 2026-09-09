"""
Unit tests for Token Alignment and Whitespace Canonicalization.
"""

import pytest
from transformers import AutoTokenizer
from token_alignment import (
    LABEL_LIST,
    ID2LABEL,
    LABEL2ID,
    canonicalize_text,
    align_spans_to_bio_tokens,
    reconstruct_spans_from_bio_predictions,
)


def test_canonicalize_text():
    raw = "Patient has   EGFR  T790M   mutation.\n\nAdministered   cisplatin   75 mg/m2.\t\t"
    cleaned = canonicalize_text(raw)
    assert cleaned == "Patient has EGFR T790M mutation. Administered cisplatin 75 mg/m2."
    assert "  " not in cleaned
    assert "\n" not in cleaned
    assert "\t" not in cleaned


def test_bio_label_schema():
    assert "O" in LABEL2ID
    assert LABEL2ID["O"] == 0
    assert "B-GENE_MUTATION" in LABEL2ID
    assert "I-GENE_MUTATION" in LABEL2ID
    assert "B-DRUG_NAME" in LABEL2ID
    assert "I-DRUG_NAME" in LABEL2ID
    assert "B-DOSAGE" in LABEL2ID
    assert "I-DOSAGE" in LABEL2ID
    assert "B-ADVERSE_EVENT" in LABEL2ID
    assert "I-ADVERSE_EVENT" in LABEL2ID
    assert len(LABEL_LIST) == 9
    assert len(ID2LABEL) == 9
    assert len(LABEL2ID) == 9


def test_align_and_reconstruct_spans():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    text = "Patient has EGFR mutation and received cisplatin 75 mg/m2."
    entities = [
        {"start": 12, "end": 25, "label": "GENE_MUTATION", "text": "EGFR mutation"},
        {"start": 39, "end": 48, "label": "DRUG_NAME", "text": "cisplatin"},
        {"start": 49, "end": 57, "label": "DOSAGE", "text": "75 mg/m2"},
    ]
    
    aligned = align_spans_to_bio_tokens(text, entities, tokenizer, max_length=64)
    assert "input_ids" in aligned
    assert "attention_mask" in aligned
    assert "labels" in aligned
    assert "offset_mapping" in aligned
    
    # Reconstruct spans using ground truth labels
    preds = [p if p != -100 else 0 for p in aligned["labels"]]
    offsets = aligned["offset_mapping"]
    
    reconstructed = reconstruct_spans_from_bio_predictions(text, token_preds=preds, offsets=offsets)
    assert len(reconstructed) == 3
    rec_labels = {r["label"] for r in reconstructed}
    assert "GENE_MUTATION" in rec_labels
    assert "DRUG_NAME" in rec_labels
    assert "DOSAGE" in rec_labels
