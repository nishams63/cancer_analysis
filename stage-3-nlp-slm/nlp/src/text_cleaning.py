"""
Conservative Clinical Text Cleaning Module for Stage 3 NLP.
Preserves numbers, units, dosages, abbreviations, and clinical negations while removing
malformed whitespace, control characters, and formatting artifacts.
"""

import re
from typing import Optional


def clean_clinical_text(text: Optional[str]) -> str:
    """
    Apply conservative cleaning to clinical text without destroying medical semantics.
    
    Preserved:
      - Numbers, decimals (e.g. 75.5 mg/m2, 1.85 mg/dL)
      - Percentages (e.g. 96%, 80%)
      - Severity grades (e.g. Grade 3, Grade 4)
      - Clinical units (e.g. mmHg, mg, mg/dL, /uL)
      - Negations (e.g. no, denies, negative for, without)
      - Medical acronyms (e.g. ECOG, SpO2, NSCLC, EGFR, KRAS)
    
    Cleaned:
      - Strips accidental HTML/XML tags
      - Strips non-printable ASCII control characters
      - Normalizes curly quotes and smart dashes to ASCII
      - Normalizes non-breaking spaces to standard space
      - Collapses consecutive whitespace and limits consecutive newlines to 2
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # 1. Remove non-printable control characters (keep \n and \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 2. Strip HTML/XML tags if accidentally present
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. Normalize non-breaking and special spaces
    text = text.replace("\xa0", " ").replace("\u200b", "").replace("\ufeff", "")

    # 4. Standardize quotes and dashes to ASCII equivalents
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("—", "-").replace("–", "-")

    # 5. Normalize whitespace within lines
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(cleaned_line)

    # 6. Recombine lines, collapsing 3+ newlines to 2
    recombined = "\n".join(cleaned_lines)
    recombined = re.sub(r"\n{3,}", "\n\n", recombined).strip()

    return recombined


def is_valid_clinical_text(text: str, min_words: int = 15, max_words: int = 400) -> bool:
    """Check whether text meets basic operational word length constraints."""
    if not text or not isinstance(text, str):
        return False
    words = text.split()
    return min_words <= len(words) <= max_words
