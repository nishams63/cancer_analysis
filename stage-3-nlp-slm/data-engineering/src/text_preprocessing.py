"""
Clinical Text Preprocessing & Normalization Module for Stage 3 NLP.
Preserves clinical negation, dosages, laboratory numbers, and abbreviations.
"""

import re
import unicodedata
from typing import Dict, Any, Tuple

# HTML tag stripper
HTML_TAG_RE = re.compile(r'<[^>]+>')

# Control character stripper (keeps newline and tab)
CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')

# Multiple horizontal spaces (leaves newlines intact)
HORIZONTAL_WHITESPACE_RE = re.compile(r'[^\S\r\n]+')

# Excessive newlines (> 2 newlines reduced to 2)
MULTIPLE_NEWLINES_RE = re.compile(r'\n{3,}')

def normalize_clinical_text(text: str) -> str:
    """
    Carefully normalize clinical text without destroying medical semantics:
    - Normalizes Unicode to NFKC.
    - Strips unintended HTML/XML tags.
    - Removes unprintable control characters.
    - Standardizes whitespace while preserving section breaks and bullet points.
    - Strictly preserves numbers, decimal points, units, negations, and acronyms.
    """
    if not isinstance(text, str):
        return ""

    # 1. Unicode NFKC normalization
    normalized = unicodedata.normalize("NFKC", text)

    # 2. Strip HTML/XML tags if any
    normalized = HTML_TAG_RE.sub("", normalized)

    # 3. Strip control characters
    normalized = CONTROL_CHAR_RE.sub("", normalized)

    # 4. Standardize quotes and dashes to standard ASCII equivalents
    normalized = normalized.replace("“", '"').replace("”", '"')
    normalized = normalized.replace("‘", "'").replace("’", "'")
    normalized = normalized.replace("—", " - ").replace("–", " - ")

    # 5. Normalize horizontal whitespace line-by-line
    lines = [HORIZONTAL_WHITESPACE_RE.sub(" ", line).strip() for line in normalized.splitlines()]
    
    # 6. Recombine and collapse excessive blank lines
    combined = "\n".join(lines)
    combined = MULTIPLE_NEWLINES_RE.sub("\n\n", combined)
    
    return combined.strip()

def compute_text_metrics(text: str) -> Dict[str, Any]:
    """
    Compute length and structural statistics for a text document.
    """
    if not text:
        return {
            "char_count": 0,
            "word_count": 0,
            "line_count": 0,
            "has_negation": False,
            "has_dosage": False,
            "has_genomic_mutations": False
        }

    words = re.findall(r'\b\S+\b', text)
    word_count = len(words)
    char_count = len(text)
    line_count = len(text.splitlines())

    # Detect presence of core clinical features
    negation_cues = ["no ", "not ", "denies", "without", "negative", "absence of", "none"]
    has_negation = any(cue in text.lower() for cue in negation_cues)
    
    has_dosage = bool(re.search(r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|mg/m2|u/l|ng/ml)\b', text, re.IGNORECASE))
    has_mutations = bool(re.search(r'\b(?:EGFR|KRAS|TP53|BRAF|ALK|ROS1|PIK3CA|MET)\b', text))

    return {
        "char_count": char_count,
        "word_count": word_count,
        "line_count": line_count,
        "has_negation": has_negation,
        "has_dosage": has_dosage,
        "has_genomic_mutations": has_mutations
    }
