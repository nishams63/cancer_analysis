"""
Context and Symptom Log Augmentation Module.
Safely varies clinical framing, patient symptom log self-reports, and observational context
while strictly preserving negation triggers, historical/resolved scope, and entity mentions.
"""

from typing import List, Dict, Any, Tuple
import random
import re
from annotation_propagation import propagate_entity_spans

# Protected polarity cues that must NEVER be removed, reversed, or added
PROTECTED_POLARITY_CUES = [
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\bdenies\b",
    r"\bdenied\b",
    r"\babsence\s+of\b",
    r"\bnegative\s+for\b",
    r"\bruled\s+out\b",
    r"\bhistory\s+of\b",
    r"\bprior\s+to\b",
    r"\bresolved\b",
    r"\bsubsided\b"
]

# Safe framing substitutions for patient symptom logs and clinical narratives
SYMPTOM_LOG_FRAMING = [
    (r"\bI am reporting my symptoms for\b", [
        "I am submitting my symptoms for",
        "Logging my daily symptoms for",
        "Reporting my oncology symptoms for"
    ]),
    (r"\bOverall I am feeling\b", [
        "In general I am feeling",
        "Overall my status today is",
        "Generally feeling"
    ]),
    (r"\bI took my prescribed medicine as directed\b", [
        "I took my prescribed medications as instructed",
        "Took all prescribed oncology medicines as directed",
        "Prescribed medicines taken per oncologist directions"
    ]),
    (r"\bWill contact the clinic if symptoms worsen\b", [
        "Will alert the oncology triage desk if symptoms worsen",
        "Plan to call the infusion clinic if symptoms intensify",
        "Will immediately contact triage team if symptoms worsen"
    ])
]


def apply_context_framing_augmentation(
    text: str,
    entities: List[Dict[str, Any]],
    rng: random.Random
) -> Tuple[str, List[Dict[str, Any]], bool, str]:
    """
    Applies conversational and contextual framing variations to patient symptom logs
    and observational context sentences.
    """
    candidates = []
    
    for pattern, options in SYMPTOM_LOG_FRAMING:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            c_start, c_end = match.start(), match.end()
            
            # Entity collision guard
            if any(max(c_start, e["start"]) < min(c_end, e["end"]) for e in entities):
                continue
                
            choice = rng.choice(options)
            if choice.lower() != match.group().lower():
                candidates.append((c_start, c_end, choice))

    if not candidates:
        return text, entities, False, "no_context_framing_matched"

    rng.shuffle(candidates)
    selected = [candidates[0]] # Apply one safe framing transformation at a time
    
    new_text, new_ents, valid = propagate_entity_spans(text, entities, selected)
    if not valid:
        return text, entities, False, "context_propagation_failed"

    return new_text, new_ents, True, "context_framing_variation"
