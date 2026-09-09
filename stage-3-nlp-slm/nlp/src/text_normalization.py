"""
Clinical Text Normalization Module for Stage 3 NLP.
Applies Unicode NFKC normalization and conservative medical unit & abbreviation standardization.
"""

import re
import unicodedata


# Controlled Unit Replacement Maps
UNIT_PATTERNS = [
    (r"\b(\d+(\.\d+)?)\s*(?:mg\s*\/\s*m\^?2|mg\s*\/\s*m2)\b", r"\1 mg/m2"),
    (r"\b(\d+(\.\d+)?)\s*(?:mg\s*\/\s*dl|mg\s*\/\s*dL)\b", r"\1 mg/dL"),
    (r"\b(\d+(\.\d+)?)\s*(?:mm\s*hg|mmHg)\b", r"\1 mmHg"),
    (r"\b(\d+(\.\d+)?)\s*(?:g\s*\/\s*dl|g\s*\/\s*dL)\b", r"\1 g/dL"),
    (r"\b(\d+(\.\d+)?)\s*mg\b", r"\1 mg"),
    (r"\b(\d+(\.\d+)?)\s*mcg\b", r"\1 mcg"),
    (r"\b(\d+(\.\d+)?)\s*%\b", r"\1%"),
]

# Standard Medical Acronym Normalization (case-insensitive boundary matching to canonical form)
ACRONYM_MAP = {
    r"\bnsclc\b": "NSCLC",
    r"\becog\b": "ECOG",
    r"\begfr\b": "EGFR",
    r"\bkras\b": "KRAS",
    r"\btp53\b": "TP53",
    r"\bbraf\b": "BRAF",
    r"\balk\b": "ALK",
    r"\bctcae\b": "CTCAE",
    r"\bspo2\b": "SpO2",
    r"\banc\b": "ANC",
    r"\balt\b": "ALT",
    r"\bast\b": "AST",
}


def standardize_clinical_units(text: str) -> str:
    """Standardize spacing and notation around clinical units and dosages."""
    for pattern, replacement in UNIT_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def standardize_abbreviations(text: str) -> str:
    """Standardize key oncology driver gene symbols and clinical scales to canonical uppercase."""
    for pattern, canonical in ACRONYM_MAP.items():
        text = re.sub(pattern, canonical, text, flags=re.IGNORECASE)
    return text


def normalize_clinical_text(text: str, form: str = "NFKC") -> str:
    """
    Perform full semantic-preserving normalization:
    1. Unicode normalization (NFKC).
    2. Clinical unit standardization.
    3. Clinical acronym casing.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 1. Unicode decomposition/composition
    normalized = unicodedata.normalize(form, text)

    # Clean zero-width spaces, byte order marks, non-breaking spaces
    normalized = normalized.replace("\u200b", "").replace("\ufeff", "").replace("\xa0", " ")

    # Standardize quotation marks and smart dashes to ASCII
    normalized = normalized.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    normalized = normalized.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    normalized = normalized.replace("—", "-").replace("–", "-")

    # 2. Units standardization
    normalized = standardize_clinical_units(normalized)

    # 3. Canonical acronym casing
    normalized = standardize_abbreviations(normalized)

    return normalized
