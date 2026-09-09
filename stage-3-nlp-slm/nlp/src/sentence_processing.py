"""
Clinical Sentence Segmentation and Tokenization Module for Stage 3 NLP.
Safely detects sentence boundaries without breaking on decimal dosages or clinical abbreviations.
"""

import re
from typing import List, Dict, Any


# Regex for protected clinical abbreviations and numeric patterns
ABBREVIATION_PATTERN = r"\b(?:Dr|Mr|Mrs|Ms|vs|e\.g|i\.e|No|Fig|tab|approx|vol|p\.o|i\.v)\."
DECIMAL_PATTERN = r"\b\d+\.\d+\b"


def split_into_sentences(text: str) -> List[Dict[str, Any]]:
    """
    Split clinical text into sentences while tracking exact character offsets.
    Protects decimals (e.g. 75.5 mg), abbreviations, and clinical headings.
    """
    if not text:
        return []

    # Temporary mask for protected dots
    protected_text = text

    # Protect decimals
    decimals = re.findall(DECIMAL_PATTERN, protected_text)
    decimal_masks = {}
    for i, dec in enumerate(decimals):
        mask = f"__DECIMAL_{i}__"
        decimal_masks[mask] = dec
        protected_text = protected_text.replace(dec, mask, 1)

    # Protect common medical abbreviations
    abbrevs = re.findall(ABBREVIATION_PATTERN, protected_text, flags=re.IGNORECASE)
    abbrev_masks = {}
    for i, abb in enumerate(abbrevs):
        mask = f"__ABBREV_{i}__"
        abbrev_masks[mask] = abb
        protected_text = protected_text.replace(abb, mask, 1)

    # Split boundaries: period, question mark, exclamation, or double newline
    # followed by whitespace or line break
    sentence_spans: List[Dict[str, Any]] = []
    
    # Use regex finditer to track exact start and end offsets in protected_text
    # Boundary: (?<=[.!?])\s+(?=[A-Z0-9\[\"]) | \n\s*\n | (?<=[.!?])$
    boundary_pattern = re.compile(r"(?:(?<=[.!?])\s+(?=[A-Z0-9\[\"])|\n{1,})")

    curr_start = 0
    sent_idx = 0

    for match in boundary_pattern.finditer(protected_text):
        end = match.start()
        raw_sent = protected_text[curr_start:end].strip()
        if raw_sent:
            # Unmask
            for mask, orig in decimal_masks.items():
                raw_sent = raw_sent.replace(mask, orig)
            for mask, orig in abbrev_masks.items():
                raw_sent = raw_sent.replace(mask, orig)

            # Find actual start/end in original text
            orig_start = text.find(raw_sent, curr_start)
            if orig_start != -1:
                orig_end = orig_start + len(raw_sent)
                sentence_spans.append({
                    "sentence_id": sent_idx,
                    "text": raw_sent,
                    "start_char": orig_start,
                    "end_char": orig_end
                })
                sent_idx += 1
        curr_start = match.end()

    # Capture remaining text after last boundary
    if curr_start < len(protected_text):
        raw_sent = protected_text[curr_start:].strip()
        if raw_sent:
            for mask, orig in decimal_masks.items():
                raw_sent = raw_sent.replace(mask, orig)
            for mask, orig in abbrev_masks.items():
                raw_sent = raw_sent.replace(mask, orig)

            orig_start = text.find(raw_sent, curr_start)
            if orig_start != -1:
                orig_end = orig_start + len(raw_sent)
                sentence_spans.append({
                    "sentence_id": sent_idx,
                    "text": raw_sent,
                    "start_char": orig_start,
                    "end_char": orig_end
                })

    return sentence_spans


def tokenize_sentence(sentence_text: str, base_offset: int = 0) -> List[Dict[str, Any]]:
    """
    Tokenize sentence into word tokens while preserving character span offsets.
    Returns list of token dicts: [{"token": str, "start_char": int, "end_char": int}]
    """
    tokens = []
    # Match alphanumeric words, including internal hyphens and slashes
    pattern = re.compile(r"\b[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*\b")
    for match in pattern.finditer(sentence_text):
        tokens.append({
            "token": match.group(),
            "start_char": base_offset + match.start(),
            "end_char": base_offset + match.end()
        })
    return tokens
