"""
Token Alignment and BIO Sequence Labeling Module for Clinical Concept Extraction.
Maps character span annotations to subword token labels and reconstructs predicted
token sequences back into validated clinical entity character spans.
"""

from typing import List, Dict, Any, Tuple, Optional
import re
import numpy as np

LABEL_LIST: List[str] = [
    "O",
    "B-ADVERSE_EVENT",
    "I-ADVERSE_EVENT",
    "B-DOSAGE",
    "I-DOSAGE",
    "B-DRUG_NAME",
    "I-DRUG_NAME",
    "B-GENE_MUTATION",
    "I-GENE_MUTATION"
]

ID2LABEL: Dict[int, str] = {i: lbl for i, lbl in enumerate(LABEL_LIST)}
LABEL2ID: Dict[str, int] = {lbl: i for i, lbl in enumerate(LABEL_LIST)}


def canonicalize_text(text: str) -> str:
    """
    Standardize text formatting by collapsing multiple whitespace characters
    and trimming leading/trailing padding. Fixes the whitespace feature sensitivity.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def align_spans_to_bio_tokens(
    text: str,
    entities: List[Dict[str, Any]],
    tokenizer,
    max_length: int = 256
) -> Dict[str, Any]:
    """
    Align ground-truth character spans [start, end, label] with subword tokens.
    Assigns:
        -100: for special tokens ([CLS], [SEP], [PAD])
        B-{LABEL}: for the first subword overlapping the entity span
        I-{LABEL}: for subsequent subwords within the entity span
        O: for non-entity tokens
    """
    tokenized = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
        return_tensors=None
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    offsets = tokenized["offset_mapping"]

    labels = []

    # Sort entities by start position to ensure deterministic matching
    sorted_ents = sorted(entities, key=lambda x: (x["start"], x["end"]))

    for i, (tok_start, tok_end) in enumerate(offsets):
        # Ignore special tokens
        if tok_start == tok_end or attention_mask[i] == 0:
            labels.append(-100)
            continue

        assigned_label = "O"
        for ent in sorted_ents:
            g_start = ent["start"]
            g_end = ent["end"]
            g_label = ent["label"]

            # Check overlap between token character slice and entity slice
            if max(tok_start, g_start) < min(tok_end, g_end):
                # If this token includes or touches the entity start, mark B-
                if tok_start <= g_start < tok_end or tok_start == g_start:
                    assigned_label = f"B-{g_label}"
                else:
                    # Check if previous token was already labeled as this entity
                    assigned_label = f"I-{g_label}"
                break

        # Fallback verification: map to ID
        label_id = LABEL2ID.get(assigned_label, 0)
        labels.append(label_id)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "offset_mapping": offsets
    }


def reconstruct_spans_from_bio_predictions(
    text: str,
    token_preds: List[int],
    token_probs: Optional[List[float]] = None,
    offsets: Optional[List[Tuple[int, int]]] = None
) -> List[Dict[str, Any]]:
    """
    Reconstruct document-level character spans from predicted BIO token label IDs.
    Merges contiguous B- and I- tokens of matching entity types.
    """
    if offsets is None or len(token_preds) != len(offsets):
        return []

    spans = []
    current_entity = None

    for i, pred_id in enumerate(token_preds):
        tok_start, tok_end = offsets[i]
        if tok_start == tok_end:  # Special token
            continue

        pred_tag = ID2LABEL.get(pred_id, "O")
        prob = token_probs[i] if token_probs is not None and i < len(token_probs) else 1.0

        if pred_tag.startswith("B-"):
            # Close previous entity if open
            if current_entity is not None:
                spans.append(current_entity)

            ent_type = pred_tag.split("-")[1]
            current_entity = {
                "start": tok_start,
                "end": tok_end,
                "label": ent_type,
                "confidence_scores": [prob]
            }

        elif pred_tag.startswith("I-"):
            ent_type = pred_tag.split("-")[1]
            if current_entity is not None and current_entity["label"] == ent_type:
                # Extend span
                current_entity["end"] = tok_end
                current_entity["confidence_scores"].append(prob)
            else:
                # Treat orphaned I- as a new B- entity
                if current_entity is not None:
                    spans.append(current_entity)
                current_entity = {
                    "start": tok_start,
                    "end": tok_end,
                    "label": ent_type,
                    "confidence_scores": [prob]
                }

        else:  # "O"
            if current_entity is not None:
                spans.append(current_entity)
                current_entity = None

    if current_entity is not None:
        spans.append(current_entity)

    # Format final entities with text snippet and aggregated confidence
    formatted_entities = []
    for s in spans:
        start_c = s["start"]
        end_c = s["end"]
        ent_text = text[start_c:end_c].strip()
        if not ent_text:
            continue

        mean_conf = float(np.mean(s["confidence_scores"])) if s.get("confidence_scores") else 1.0
        formatted_entities.append({
            "start": start_c,
            "end": end_c,
            "label": s["label"],
            "text": ent_text,
            "confidence": round(mean_conf, 4)
        })

    return formatted_entities
