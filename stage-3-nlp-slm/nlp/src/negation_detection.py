"""
Clinical Negation and Context Attribution Module for Stage 3 NLP.
Implements a deterministic, rule-based NegEx-style algorithm to classify clinical concepts
into AFFIRMED, NEGATED, HISTORICAL, or RESOLVED categories.
"""

import re
from typing import List, Dict, Any, Tuple


# Pre-negation triggers that project negation forward within a scope window
PRE_NEGATION_TRIGGERS = [
    r"\bno\s+evidence\s+of\b",
    r"\bnegative\s+for\b",
    r"\bruled\s+out\b",
    r"\babsence\s+of\b",
    r"\bdenies\b",
    r"\bdenied\b",
    r"\bwithout\b",
    r"\bno\b",
    r"\bnot\b",
]

# Post-negation triggers that project negation backward
POST_NEGATION_TRIGGERS = [
    r"\bunlikely\b",
    r"\babsent\b",
    r"\bresolved\b",
    r"\bsubsided\b",
    r"\bnegative\b",
    r"\bfree\b"
]

# Historical context triggers
HISTORICAL_TRIGGERS = [
    r"\bhistory\s+of\b",
    r"\bpreviously\s+treated\s+with\b",
    r"\bprior\s+history\s+of\b",
    r"\bprior\b",
    r"\bstatus\s+post\b",
    r"\bpast\s+medical\s+history\b"
]

# Pseudo-negations that look like negation but should NOT negate concepts
PSEUDO_NEGATIONS = [
    r"\bno\s+change\b",
    r"\bno\s+increase\b",
    r"\bnot\s+only\b",
    r"\bno\s+doubt\b"
]

# Scope termination boundaries (conjunctions and punctuation that terminate scope)
SCOPE_TERMINATORS = [r"\bbut\b", r"\bhowever\b", r"\balthough\b", r"\bnevertheless\b", r";", r":"]


def _find_trigger_spans(text: str, patterns: List[str]) -> List[Tuple[int, int, str]]:
    """Find all matching trigger spans in text (start_char, end_char, matched_str)."""
    spans = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            spans.append((m.start(), m.end(), m.group()))
    return spans


def resolve_concept_polarity(sentence_text: str, concept_start: int, concept_end: int,
                             max_scope_words: int = 6) -> str:
    """
    Determine the clinical polarity of a concept within a sentence.
    Returns: 'NEGATED', 'HISTORICAL', 'RESOLVED', or 'AFFIRMED'.
    """
    sent_lower = sentence_text.lower()
    c_start = concept_start
    c_end = concept_end

    # Find all pseudo-negation spans to ignore triggers inside them
    pseudo_spans = _find_trigger_spans(sent_lower, PSEUDO_NEGATIONS)

    # 1. Check if concept itself is part of pseudo-negation
    for p_start, p_end, _ in pseudo_spans:
        if p_start <= c_start <= p_end or p_start <= c_end <= p_end:
            return "AFFIRMED"

    # 2. Check for Historical triggers preceding concept
    for h_start, h_end, _ in _find_trigger_spans(sent_lower, HISTORICAL_TRIGGERS):
        if h_end <= c_start:
            between = sent_lower[h_end:c_start].strip()
            words_between = between.split()
            has_terminator = any(re.search(term, between) for term in SCOPE_TERMINATORS)
            if len(words_between) <= max_scope_words and not has_terminator:
                return "HISTORICAL"

    # 3. Check for Pre-Negation triggers preceding concept (excluding pseudo-negation spans)
    for n_start, n_end, _ in _find_trigger_spans(sent_lower, PRE_NEGATION_TRIGGERS):
        # Skip if this trigger is inside a pseudo-negation (e.g. 'no' inside 'no change')
        is_pseudo = any(p_start <= n_start and n_end <= p_end for p_start, p_end, _ in pseudo_spans)
        if is_pseudo:
            continue

        if n_end <= c_start:
            between = sent_lower[n_end:c_start].strip()
            words_between = between.split()
            has_terminator = any(re.search(term, between) for term in SCOPE_TERMINATORS)
            if len(words_between) <= max_scope_words and not has_terminator:
                return "NEGATED"

    # 4. Check for Post-Negation triggers following concept
    for n_start, n_end, term_str in _find_trigger_spans(sent_lower, POST_NEGATION_TRIGGERS):
        if c_end <= n_start:
            between = sent_lower[c_end:n_start].strip()
            words_between = between.split()
            has_terminator = any(re.search(term, between) for term in SCOPE_TERMINATORS)
            if len(words_between) <= 3 and not has_terminator:
                if term_str.lower() in ["resolved", "subsided"]:
                    return "RESOLVED"
                return "NEGATED"

    # Default if no negation/historical triggers apply
    return "AFFIRMED"


def detect_negations_in_sentence(sentence_text: str, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Attach polarity status ('AFFIRMED', 'NEGATED', 'HISTORICAL', 'RESOLVED')
    to each concept dictionary in the sentence.
    """
    attributed_concepts = []
    for c in concepts:
        c_copy = dict(c)
        status = resolve_concept_polarity(
            sentence_text,
            c["start_char"],
            c["end_char"]
        )
        c_copy["polarity"] = status
        attributed_concepts.append(c_copy)
    return attributed_concepts
