"""
Clinical Negation-Scope and Negation-Flip Audit Module for Stage 4 EDA.
Implements clinical polarity verification using Stage 3 NegEx rules,
detects safety-critical negation flips between source notes and targets,
and exports granular audit cases to Parquet per Section 13.
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np


# Production clinical negation triggers adapted from Stage 3 NLP
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

POST_NEGATION_TRIGGERS = [
    r"\bunlikely\b",
    r"\babsent\b",
    r"\bresolved\b",
    r"\bsubsided\b",
    r"\bnegative\b",
    r"\bfree\b",
]

PSEUDO_NEGATIONS = [
    r"\bno\s+change\b",
    r"\bno\s+increase\b",
    r"\bnot\s+only\b",
    r"\bno\s+doubt\b",
]

SCOPE_TERMINATORS = [r"\bbut\b", r"\bhowever\b", r"\balthough\b", r"\bnevertheless\b", r";", r":"]


class NegationAnalyzer:
    """Audits clinical polarity consistency and identifies critical negation inversion risks."""

    def __init__(self, max_scope_words: int = 6):
        self.max_scope_words = max_scope_words

    def is_concept_negated_in_sentence(self, sentence: str, concept: str) -> bool:
        """Determines if a concept is negated in a given sentence."""
        sent_lower = sentence.lower()
        concept_lower = concept.lower()
        if concept_lower not in sent_lower:
            return False

        # Find concept start/end indices
        m = re.search(r"\b" + re.escape(concept_lower) + r"\b", sent_lower)
        if not m:
            return False
        c_start, c_end = m.start(), m.end()

        # Check pseudo-negations
        for p_pat in PSEUDO_NEGATIONS:
            for pm in re.finditer(p_pat, sent_lower):
                if pm.start() <= c_start <= pm.end() or pm.start() <= c_end <= pm.end():
                    return False

        # Check pre-negation triggers
        for n_pat in PRE_NEGATION_TRIGGERS:
            for nm in re.finditer(n_pat, sent_lower):
                # Ensure trigger is not inside pseudo-negation
                if any(pm.start() <= nm.start() and nm.end() <= pm.end()
                       for p_pat in PSEUDO_NEGATIONS for pm in re.finditer(p_pat, sent_lower)):
                    continue
                if nm.end() <= c_start:
                    between = sent_lower[nm.end():c_start].strip()
                    words_between = between.split()
                    has_terminator = any(re.search(term, between) for term in SCOPE_TERMINATORS)
                    if len(words_between) <= self.max_scope_words and not has_terminator:
                        return True

        # Check post-negation triggers
        for post_pat in POST_NEGATION_TRIGGERS:
            for pm in re.finditer(post_pat, sent_lower):
                if c_end <= pm.start():
                    between = sent_lower[c_end:pm.start()].strip()
                    words_between = between.split()
                    has_terminator = any(re.search(term, between) for term in SCOPE_TERMINATORS)
                    if len(words_between) <= self.max_scope_words and not has_terminator:
                        return True

        return False

    def find_source_negated_concepts(self, text: str, candidate_concepts: List[str]) -> List[Tuple[str, str]]:
        """
        Extracts tuples of (negated_concept, context_sentence) from clinical note text.
        """
        negated = []
        sentences = re.split(r"[.\n]+", text)
        for sent in sentences:
            sent_str = sent.strip()
            if not sent_str:
                continue
            for concept in candidate_concepts:
                if self.is_concept_negated_in_sentence(sent_str, concept):
                    negated.append((concept, sent_str))
        return negated

    # Standard clinical toxicities and symptoms grounded in oncology NLP
    CLINICAL_SYMPTOMS_AND_TOXICITIES = [
        "neutropenia", "febrile neutropenia", "thrombocytopenia", "anemia", "leukopenia",
        "dyspnea", "shortness of breath", "cough", "pneumonitis",
        "fatigue", "asthenia", "malaise", "lethargy",
        "nausea", "vomiting", "diarrhea", "constipation",
        "rash", "pruritus", "dermatitis", "alopecia",
        "neuropathy", "paresthesia", "numbness", "tingling",
        "transaminases", "hepatotoxicity",
        "nephrotoxicity", "acute kidney injury",
        "cardiotoxicity", "arrhythmia", "heart failure",
        "chest pain", "respiratory distress", "adverse toxicities"
    ]

    def audit_negation_in_dataset(self, df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Scans all records for negated conditions in the source clinical notes and audits
        how they are represented in the generated target risk, findings, and actions.
        """
        review_cases = []
        total_negated_entities = 0
        correctly_preserved = 0
        negation_flips = 0

        # Build combined candidate concept lexicon
        candidate_concepts = set(self.CLINICAL_SYMPTOMS_AND_TOXICITIES)
        for ae_list in df["ner_adverse_events"].dropna():
            if isinstance(ae_list, (list, np.ndarray)):
                for ae in ae_list:
                    ae_clean = str(ae).strip().lower()
                    if ae_clean.startswith("no "):
                        ae_clean = ae_clean[3:].strip()
                    if ae_clean and ae_clean not in ("none/unknown", "none"):
                        candidate_concepts.add(ae_clean)

        for idx, row in df.iterrows():
            note = str(row.get("clinical_note", ""))
            t_risk = str(row.get("target_risk", ""))
            t_kf = str(row.get("target_key_finding", ""))
            t_act = str(row.get("target_action", ""))
            target_combined = f"{t_risk} {t_kf} {t_act}".lower()

            negated_in_source = self.find_source_negated_concepts(note, list(candidate_concepts))

            for concept, sent_ctx in negated_in_source:
                total_negated_entities += 1
                concept_lower = concept.lower()

                # Check representation in target:
                if concept_lower in target_combined:
                    # Check if negated in target
                    is_neg_in_target = (
                        f"no {concept_lower}" in target_combined
                        or f"no evidence of {concept_lower}" in target_combined
                        or f"denies {concept_lower}" in target_combined
                        or f"without {concept_lower}" in target_combined
                        or f"negative for {concept_lower}" in target_combined
                    )

                    # Check if target actively affirms it as a hazard
                    is_affirmed_hazard = (
                        f"increased {concept_lower}" in target_combined
                        or f"developed {concept_lower}" in target_combined
                        or f"hazard associated with" in target_combined and concept_lower in t_risk.lower()
                    )

                    if is_affirmed_hazard and not is_neg_in_target:
                        negation_flips += 1
                        review_cases.append({
                            "patient_id": row.get("patient_id", ""),
                            "note_id": row.get("note_id", ""),
                            "source_text": sent_ctx,
                            "negated_entity": concept,
                            "target_text": f"Risk: {t_risk} | Finding: {t_kf}",
                            "detected_issue": "NEGATION_FLIP_AFFIRMED_HAZARD"
                        })
                    else:
                        correctly_preserved += 1
                else:
                    # Omitted or safely absent from active target assertions
                    correctly_preserved += 1

        flip_rate = float(round((negation_flips / total_negated_entities), 4)) if total_negated_entities > 0 else 0.0

        metrics = {
            "total_negated_entities_detected": total_negated_entities,
            "correctly_preserved": correctly_preserved,
            "negation_flips": negation_flips,
            "negation_flip_rate": flip_rate,
            "audit_cases_count": len(review_cases)
        }

        review_df = pd.DataFrame(
            review_cases,
            columns=["patient_id", "note_id", "source_text", "negated_entity", "target_text", "detected_issue"]
        )

        return metrics, review_df
