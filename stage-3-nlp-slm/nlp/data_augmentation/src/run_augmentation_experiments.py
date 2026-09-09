"""
Comprehensive Benchmark Experiment Runner for Stage 3 Augmented Datasets.
Trains and evaluates:
1. Baseline A (TF-IDF + Structured Features + LR)
2. MiniLM Hybrid (Contextual MiniLM Embeddings + Structured Features + LR)
3. Train Span Lexicon (Train-Derived Lexicon + Negation Scoping)
across all 5 dataset versions on the official UNCHANGED VALIDATION partition:
- Config A: Original Training Data (4,261 docs)
- Config B: Original + 25% Augmentation (5,326 docs)
- Config C: Original + 50% Augmentation (6,391 docs)
- Config D: Original + 100% Augmentation (8,522 docs)
- Config E: Targeted Class-Balanced Augmentation (5,761 docs)

Locked-test partition is STRICTLY NEVER ACCESSED.
"""

import sys
from pathlib import Path
import json
import time
import re
import collections
import copy
import psutil
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

CUR_DIR = Path(__file__).resolve().parent
REPO_ROOT = CUR_DIR.parent.parent.parent.parent
NLP_SRC = REPO_ROOT / "stage-3-nlp-slm" / "nlp" / "src"
BENCH_SRC = REPO_ROOT / "stage-3-nlp-slm" / "nlp" / "benchmarking" / "src"
DATA_DIR = CUR_DIR.parent / "data" / "augmented_train"
RESULTS_DIR = CUR_DIR.parent / "results"
INTERMEDIATE_DIR = CUR_DIR.parent / "data" / "intermediate"

for p in [str(NLP_SRC), str(BENCH_SRC), str(CUR_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from protocol import LABELS, span_metrics, classification
from feature_extraction import create_negation_scoped_text, extract_structured_concept_features
from label_preparation import TargetLabelManager
from negation_detection import resolve_concept_polarity

CHECKPOINT = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TrainSpanLexicon:
    """Train-only Span Lexicon that extracts longest non-overlapping phrase matches."""
    def __init__(self, terms):
        self.terms = terms
        self.lookup = {t["phrase"]: t["label"] for t in terms}
        patterns = [r"\s+".join(map(re.escape, t["phrase"].split()))
                    for t in sorted(terms, key=lambda t: -len(t["phrase"])) if t["phrase"].strip()]
        self.pattern = re.compile(r"(?<!\w)(?:" + "|".join(patterns) + r")(?!\w)", re.I) if patterns else None

    def extract(self, text):
        if not self.pattern:
            return []
        output = []
        for match in self.pattern.finditer(text):
            key = " ".join(match.group().lower().split())
            if key in self.lookup:
                output.append({
                    "start": match.start(),
                    "end": match.end(),
                    "label": self.lookup[key],
                    "text": match.group(),
                    "polarity": resolve_concept_polarity(text, match.start(), match.end())
                })
        return output


def get_or_create_features(df: pd.DataFrame, cache_prefix: str) -> Tuple[List[str], np.ndarray]:
    """Cache negation-scoped texts and structured numeric features to avoid redundant regex parsing."""
    scoped_cache = INTERMEDIATE_DIR / f"{cache_prefix}.scoped.json"
    struct_cache = INTERMEDIATE_DIR / f"{cache_prefix}.struct.npy"

    numeric_cols = [
        "word_count", "char_count", "total_concepts",
        "affirmed_concepts", "negated_concepts", "historical_concepts",
        "drug_mentions", "mutation_mentions", "dosage_mentions",
        "adverse_event_mentions", "has_grade_3_4", "has_critical_symptom"
    ]

    if scoped_cache.exists() and struct_cache.exists():
        with open(scoped_cache, "r", encoding="utf-8") as f:
            scoped_texts = json.load(f)
        struct_features = np.load(struct_cache)
        return scoped_texts, struct_features

    print(f"Extracting features for {cache_prefix} ({len(df)} docs)...", flush=True)
    scoped_texts = [create_negation_scoped_text(t) for t in df["text"]]
    struct_df = extract_structured_concept_features(df)
    struct_features = struct_df[numeric_cols].values.astype(float)

    with open(scoped_cache, "w", encoding="utf-8") as f:
        json.dump(scoped_texts, f)
    np.save(struct_cache, struct_features)

    return scoped_texts, struct_features


def run_experiments():
    print("=" * 80, flush=True)
    print("STAGE 3 CLINICAL NLP: AUGMENTATION BENCHMARK EXPERIMENTS", flush=True)
    print("=" * 80, flush=True)

    # 1. Load official validation split (STRICTLY READ-ONLY)
    val_path = REPO_ROOT / "stage-3-nlp-slm" / "data-engineering" / "data" / "processed" / "validation.parquet"
    df_val = pd.read_parquet(val_path)
    print(f"Loaded VALIDATION dataset: {len(df_val)} documents (150 patients). ZERO modifications allowed.", flush=True)
    gold_entities = [json.loads(e) for e in df_val["ner_entities"]]

    # Load validation cached features & embeddings
    val_scoped, val_struct = get_or_create_features(df_val, "val")
    val_emb = np.load(INTERMEDIATE_DIR / "val.emb.npy")

    dataset_configs = [
        ("Config A (Original)", "train_original.parquet"),
        ("Config B (+25% Aug)", "train_augmented_25.parquet"),
        ("Config C (+50% Aug)", "train_augmented_50.parquet"),
        ("Config D (+100% Aug)", "train_augmented_100.parquet"),
        ("Config E (Targeted Balanced)", "train_augmented_targeted.parquet")
    ]

    results_all = {}
    summary_table = []

    urgency_labels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    hazard_labels = ["NONE", "HEMATOLOGIC", "HEPATIC", "RENAL", "CARDIAC", "PULMONARY", "NEUROPATHIC", "DERMATOLOGIC"]

    for config_name, filename in dataset_configs:
        print("\n" + "#" * 60, flush=True)
        print(f"RUNNING EXPERIMENT: {config_name} ({filename})", flush=True)
        print("#" * 60, flush=True)

        df_train = pd.read_parquet(DATA_DIR / filename)
        train_count = len(df_train)
        unique_pts = int(df_train["patient_id"].nunique())
        print(f"Training instances: {train_count} | Unique patients: {unique_pts}", flush=True)

        # Load train cached features & embeddings
        train_scoped, train_struct = get_or_create_features(df_train, filename)
        train_emb = np.load(INTERMEDIATE_DIR / f"{filename}.emb.npy")

        config_result = {
            "dataset_config": config_name,
            "filename": filename,
            "training_instances": train_count,
            "unique_patients": unique_pts,
            "models": {}
        }

        # -------------------------------------------------------------
        # MODEL 1: Baseline A (TF-IDF + Structured Features + LR)
        # -------------------------------------------------------------
        print("\n--> Training Baseline A (TF-IDF + Structured + LR)...", flush=True)
        start_t = time.perf_counter()

        tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.95)
        X_train_tfidf = tfidf.fit_transform(train_scoped)
        X_val_tfidf = tfidf.transform(val_scoped)

        scaler_base = StandardScaler()
        X_train_struct_scaled = scaler_base.fit_transform(train_struct)
        X_val_struct_scaled = scaler_base.transform(val_struct)

        X_train_base = hstack([X_train_tfidf, X_train_struct_scaled]).tocsr()
        X_val_base = hstack([X_val_tfidf, X_val_struct_scaled]).tocsr()

        urg_lr = LogisticRegression(class_weight="balanced", C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
        haz_lr = LogisticRegression(class_weight="balanced", C=0.5, solver="lbfgs", max_iter=1000, random_state=42)
        urg_lr.fit(X_train_base, df_train["urgency_level"])
        haz_lr.fit(X_train_base, df_train["hazard_type"])
        train_time_base = time.perf_counter() - start_t

        start_inf = time.perf_counter()
        pred_urg_base = urg_lr.predict(X_val_base)
        pred_haz_base = haz_lr.predict(X_val_base)
        inf_time_base = (time.perf_counter() - start_inf) / len(df_val)

        urg_metrics_base = classification(df_val["urgency_level"], pred_urg_base, urgency_labels)
        haz_metrics_base = classification(df_val["hazard_type"], pred_haz_base, hazard_labels)

        config_result["models"]["baseline_a"] = {
            "training_seconds": round(train_time_base, 2),
            "inference_seconds_per_doc": round(inf_time_base, 5),
            "urgency": urg_metrics_base,
            "hazard": haz_metrics_base
        }
        print(f"Baseline A - Urgency Macro F1: {urg_metrics_base['macro_f1']:.4f}, Critical Recall: {urg_metrics_base['critical_recall']:.4f}, Hazard Macro F1: {haz_metrics_base['macro_f1']:.4f}", flush=True)

        # -------------------------------------------------------------
        # MODEL 2: MiniLM Hybrid (MiniLM Embeddings + Structured + LR)
        # -------------------------------------------------------------
        print("\n--> Training MiniLM Hybrid (Embeddings + Structured + LR)...", flush=True)
        start_t = time.perf_counter()

        scaler_hybrid = StandardScaler()
        train_struct_scaled = scaler_hybrid.fit_transform(train_struct)
        val_struct_scaled = scaler_hybrid.transform(val_struct)

        X_train_hybrid = np.hstack([train_emb, train_struct_scaled])
        X_val_hybrid = np.hstack([val_emb, val_struct_scaled])

        urg_hybrid = LogisticRegression(class_weight="balanced", C=1.0, solver="lbfgs", max_iter=1500, random_state=42)
        haz_hybrid = LogisticRegression(class_weight="balanced", C=0.5, solver="lbfgs", max_iter=1500, random_state=42)
        urg_hybrid.fit(X_train_hybrid, df_train["urgency_level"])
        haz_hybrid.fit(X_train_hybrid, df_train["hazard_type"])
        train_time_hybrid = time.perf_counter() - start_t

        start_inf = time.perf_counter()
        pred_urg_hybrid = urg_hybrid.predict(X_val_hybrid)
        pred_haz_hybrid = haz_hybrid.predict(X_val_hybrid)
        inf_time_hybrid = (time.perf_counter() - start_inf) / len(df_val)

        urg_metrics_hybrid = classification(df_val["urgency_level"], pred_urg_hybrid, urgency_labels)
        haz_metrics_hybrid = classification(df_val["hazard_type"], pred_haz_hybrid, hazard_labels)

        config_result["models"]["minilm_hybrid"] = {
            "training_seconds": round(train_time_hybrid, 2),
            "inference_seconds_per_doc": round(inf_time_hybrid, 5),
            "urgency": urg_metrics_hybrid,
            "hazard": haz_metrics_hybrid
        }
        print(f"MiniLM Hybrid - Urgency Macro F1: {urg_metrics_hybrid['macro_f1']:.4f}, Critical Recall: {urg_metrics_hybrid['critical_recall']:.4f}, Hazard Macro F1: {haz_metrics_hybrid['macro_f1']:.4f}", flush=True)

        # -------------------------------------------------------------
        # MODEL 3: Train Span Lexicon (Entity Extraction from Train Only)
        # -------------------------------------------------------------
        print("\n--> Inducing Train Span Lexicon...", flush=True)
        start_t = time.perf_counter()
        counts = collections.defaultdict(collections.Counter)
        for row in df_train.itertuples():
            ents = json.loads(row.ner_entities) if isinstance(row.ner_entities, str) else row.ner_entities
            for e in ents:
                phrase = " ".join(row.text[e["start"]:e["end"]].lower().split())
                if phrase:
                    counts[phrase][e["label"]] += 1
        terms = [{"phrase": p, "label": c.most_common(1)[0][0], "support": sum(c.values())}
                 for p, c in counts.items()]
        lexicon_model = TrainSpanLexicon(terms)
        lexicon_train_time = time.perf_counter() - start_t

        start_inf = time.perf_counter()
        pred_entities = [lexicon_model.extract(t) for t in df_val["text"]]
        inf_time_lex = (time.perf_counter() - start_inf) / len(df_val)

        ner_exact = span_metrics(gold_entities, pred_entities, relaxed=False)
        ner_relaxed = span_metrics(gold_entities, pred_entities, relaxed=True)

        config_result["models"]["train_span_lexicon"] = {
            "learned_phrase_count": len(terms),
            "training_seconds": round(lexicon_train_time, 2),
            "inference_seconds_per_doc": round(inf_time_lex, 5),
            "ner_exact": ner_exact,
            "ner_relaxed": ner_relaxed
        }
        print(f"Train Span Lexicon - Phrases: {len(terms)}, Exact F1: {ner_exact['micro']['f1']:.4f}, Relaxed F1: {ner_relaxed['micro']['f1']:.4f}", flush=True)

        results_all[config_name] = config_result

        summary_table.append({
            "Dataset Version": config_name,
            "Training Instances": train_count,
            "Unique Patients": unique_pts,
            "Baseline Urgency F1": urg_metrics_base["macro_f1"],
            "Baseline Critical Recall": urg_metrics_base["critical_recall"],
            "Baseline Hazard F1": haz_metrics_base["macro_f1"],
            "Hybrid Urgency F1": urg_metrics_hybrid["macro_f1"],
            "Hybrid Critical Recall": urg_metrics_hybrid["critical_recall"],
            "Hybrid Hazard F1": haz_metrics_hybrid["macro_f1"],
            "NER Exact F1": ner_exact["micro"]["f1"],
            "NER Relaxed F1": ner_relaxed["micro"]["f1"]
        })

    # Save complete JSON results
    with open(RESULTS_DIR / "experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=2)

    df_summary = pd.DataFrame(summary_table)
    df_summary.to_csv(RESULTS_DIR / "summary_comparison.csv", index=False)
    print("\n" + "=" * 80, flush=True)
    print("FINAL DATA SIZE / AUGMENTATION EXPERIMENT SUMMARY TABLE:", flush=True)
    print("=" * 80, flush=True)
    print(df_summary.to_string(index=False), flush=True)

    print("\nExperiments completed successfully!", flush=True)


if __name__ == "__main__":
    run_experiments()
