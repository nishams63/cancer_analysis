"""
Clinical Negation & Context Diagnostic Evaluation Module for Stage 3 Clinical NLP.
Evaluates deterministic NegEx-style scoping behavior using controlled diagnostic suites,
scope distance tests, punctuation termination checks, and pseudo-negation discrimination.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import sys

from config import NLP_ROOT_DIR

nlp_src_dir = (NLP_ROOT_DIR / "src").resolve()
if str(nlp_src_dir) not in sys.path:
    sys.path.append(str(nlp_src_dir))

from negation_detection import resolve_concept_polarity
from clinical_concepts import extract_clinical_concepts

# Controlled diagnostic benchmark suite for systematic rule testing
DIAGNOSTIC_CASES: List[Dict[str, Any]] = [
    # 1. Affirmative mentions
    {
        "category": "affirmative",
        "sentence": "Patient reports severe nausea and acute fatigue following chemotherapy.",
        "concept": "nausea",
        "expected_polarity": "AFFIRMED"
    },
    {
        "category": "affirmative",
        "sentence": "Observed moderate rash on bilateral lower extremities.",
        "concept": "rash",
        "expected_polarity": "AFFIRMED"
    },
    # 2. Pre-negation cues
    {
        "category": "pre_negation",
        "sentence": "Patient denies dyspnea or shortness of breath today.",
        "concept": "dyspnea",
        "expected_polarity": "NEGATED"
    },
    {
        "category": "pre_negation",
        "sentence": "Physical examination confirms no evidence of pneumonitis.",
        "concept": "pneumonitis",
        "expected_polarity": "NEGATED"
    },
    {
        "category": "pre_negation",
        "sentence": "Labs indicate patient is negative for neutropenia.",
        "concept": "neutropenia",
        "expected_polarity": "NEGATED"
    },
    {
        "category": "pre_negation",
        "sentence": "Patient proceeded with treatment without acute adverse toxicities.",
        "concept": "acute adverse toxicities",
        "expected_polarity": "NEGATED"
    },
    # 3. Post-negation cues
    {
        "category": "post_negation",
        "sentence": "Cardiotoxicity was deemed unlikely given normal echocardiogram.",
        "concept": "cardiotoxicity",
        "expected_polarity": "NEGATED"
    },
    {
        "category": "post_negation",
        "sentence": "Nausea is currently absent.",
        "concept": "nausea",
        "expected_polarity": "NEGATED"
    },
    # 4. Historical context
    {
        "category": "historical",
        "sentence": "Patient with history of breast cancer presenting for consultation.",
        "concept": "breast cancer",
        "expected_polarity": "HISTORICAL"
    },
    {
        "category": "historical",
        "sentence": "Patient was previously treated with cisplatin.",
        "concept": "cisplatin",
        "expected_polarity": "HISTORICAL"
    },
    {
        "category": "historical",
        "sentence": "Status post docetaxel therapy with neuropathy.",
        "concept": "docetaxel",
        "expected_polarity": "HISTORICAL"
    },
    # 5. Resolved context
    {
        "category": "resolved",
        "sentence": "Prior chemotherapy-induced vomiting has resolved completely.",
        "concept": "vomiting",
        "expected_polarity": "RESOLVED"
    },
    {
        "category": "resolved",
        "sentence": "Acute transaminitis has subsided.",
        "concept": "acute transaminitis",
        "expected_polarity": "RESOLVED"
    },
    # 6. Scope termination boundaries
    {
        "category": "scope_termination",
        "sentence": "Patient denies nausea, but reports severe fatigue.",
        "concept": "fatigue",
        "expected_polarity": "AFFIRMED"
    },
    {
        "category": "scope_termination",
        "sentence": "No diarrhea observed; however, patient complains of rash.",
        "concept": "rash",
        "expected_polarity": "AFFIRMED"
    },
    {
        "category": "scope_termination",
        "sentence": "Without neuropathy; nevertheless, lethargy persists.",
        "concept": "lethargy",
        "expected_polarity": "AFFIRMED"
    },
    # 7. Pseudo-negation handling
    {
        "category": "pseudo_negation",
        "sentence": "Patient exhibits no change in chronic fatigue levels.",
        "concept": "fatigue",
        "expected_polarity": "AFFIRMED"
    },
    {
        "category": "pseudo_negation",
        "sentence": "Patient presented with not only nausea but also diarrhea.",
        "concept": "nausea",
        "expected_polarity": "AFFIRMED"
    },
    {
        "category": "pseudo_negation",
        "sentence": "There is no increase in peripheral neuropathy severity.",
        "concept": "neuropathy",
        "expected_polarity": "AFFIRMED"
    },
    # 8. Scope window distance
    {
        "category": "scope_distance",
        "sentence": "No acute distress noted upon general physical examination while reviewing previous oncology records indicating stable neuropathy.",
        "concept": "neuropathy",
        "expected_polarity": "AFFIRMED"  # More than 6 tokens away from 'No'
    }
]


def evaluate_diagnostic_negation_suite() -> Dict[str, Any]:
    """
    Run the deterministic diagnostic negation suite against synthetic clinical cases.
    Returns per-category and overall accuracy, along with detailed test results.
    """
    results = []
    correct_count = 0
    cat_counts = {}

    for case in DIAGNOSTIC_CASES:
        sent = case["sentence"]
        c_str = case["concept"]
        exp_pol = case["expected_polarity"]
        cat = case["category"]

        c_start = sent.lower().find(c_str.lower())
        c_end = c_start + len(c_str) if c_start != -1 else 0

        pred_pol = resolve_concept_polarity(sent, c_start, c_end)
        is_correct = (pred_pol == exp_pol)
        if is_correct:
            correct_count += 1

        if cat not in cat_counts:
            cat_counts[cat] = {"total": 0, "correct": 0}
        cat_counts[cat]["total"] += 1
        if is_correct:
            cat_counts[cat]["correct"] += 1

        results.append({
            "category": cat,
            "sentence": sent,
            "concept": c_str,
            "expected": exp_pol,
            "predicted": pred_pol,
            "is_correct": is_correct
        })

    cat_summary = {}
    for cat, d in cat_counts.items():
        acc = d["correct"] / d["total"] if d["total"] > 0 else 0.0
        cat_summary[cat] = {
            "total": d["total"],
            "correct": d["correct"],
            "accuracy": round(acc, 4)
        }

    overall_acc = correct_count / len(DIAGNOSTIC_CASES)
    return {
        "total_test_cases": len(DIAGNOSTIC_CASES),
        "total_correct": correct_count,
        "overall_accuracy": round(overall_acc, 4),
        "category_performance": cat_summary,
        "detailed_results": results
    }


def evaluate_dataset_polarity_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate the natural distribution of concept polarities extracted across a dataset partition.
    """
    polarity_counts = {"AFFIRMED": 0, "NEGATED": 0, "HISTORICAL": 0, "RESOLVED": 0}
    total_entities = 0

    for _, row in df.iterrows():
        entities = extract_clinical_concepts(row["text"], assign_polarity=True)
        for ent in entities:
            pol = ent.get("polarity", "AFFIRMED")
            if pol in polarity_counts:
                polarity_counts[pol] += 1
            total_entities += 1

    ratios = {
        k: round(v / total_entities, 4) if total_entities > 0 else 0.0
        for k, v in polarity_counts.items()
    }

    return {
        "total_entities_evaluated": total_entities,
        "polarity_counts": polarity_counts,
        "polarity_ratios": ratios
    }
