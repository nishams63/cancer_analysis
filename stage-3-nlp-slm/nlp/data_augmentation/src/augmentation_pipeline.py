"""
Core End-to-End Data Augmentation Pipeline.
Generates, quality-filters, and persists train-only augmented datasets:
- Original (4,261 docs)
- Original + 25% (+1,065 docs = 5,326 total)
- Original + 50% (+2,130 docs = 6,391 total)
- Original + 100% (+4,261 docs = 8,522 total)
- Targeted Class-Balanced (boosting rare urgency and hazard classes)
"""

from typing import List, Dict, Any, Tuple, Optional
import os
import sys
import json
import time
import random
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

# Add src directories to sys.path
CUR_DIR = Path(__file__).resolve().parent
if str(CUR_DIR) not in sys.path:
    sys.path.insert(0, str(CUR_DIR))

from entity_preserving_augmentation import augment_document_preserving_entities
from quality_control import QualityControlGate
from duplicate_detection import DuplicateRegistry, compute_canonical_hash
from leakage_checks import verify_patient_split_isolation

CONFIG_PATH = CUR_DIR.parent / "configs" / "augmentation_config.yaml"


def load_yaml_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_dataset_diversity(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute rich lexical, clinical, and entity diversity metrics."""
    words_list = []
    total_tokens = 0
    all_vocab = set()
    lengths = []

    entity_counts = collections.Counter()
    for row in df.itertuples():
        text = row.text
        tokens = [t.lower() for t in text.split()]
        total_tokens += len(tokens)
        all_vocab.update(tokens)
        lengths.append(len(tokens))
        
        raw_ents = row.ner_entities
        ents = json.loads(raw_ents) if isinstance(raw_ents, str) else raw_ents
        for e in ents:
            entity_counts[e["label"]] += 1

    unique_patients = int(df["patient_id"].nunique())
    unique_encounters = int(df["encounter_id"].nunique())
    ttr = len(all_vocab) / total_tokens if total_tokens > 0 else 0.0

    return {
        "total_documents": len(df),
        "unique_patients": unique_patients,
        "unique_encounters": unique_encounters,
        "vocabulary_size": len(all_vocab),
        "total_tokens": total_tokens,
        "type_token_ratio": round(ttr, 6),
        "mean_word_length": round(float(np.mean(lengths)), 2),
        "median_word_length": round(float(np.median(lengths)), 2),
        "std_word_length": round(float(np.std(lengths)), 2),
        "urgency_distribution": df["urgency_level"].value_counts().to_dict(),
        "hazard_distribution": df["hazard_type"].value_counts().to_dict(),
        "ner_entity_distribution": dict(entity_counts)
    }


import collections

def generate_augmented_candidates(
    df_train: pd.DataFrame,
    target_count: int,
    strategy_mode: str,
    qc_gate: QualityControlGate,
    duplicate_registry: DuplicateRegistry,
    rng: random.Random
) -> List[Dict[str, Any]]:
    """
    Generate target_count high-quality augmented candidates passing the 10-point QC gate.
    """
    train_patients = set(df_train["patient_id"])
    augmented_rows = []
    
    # Priority sampling for targeted mode
    if strategy_mode == "targeted":
        # Over-sample rare urgency (CRITICAL, HIGH, MEDIUM) and rare hazard toxicities
        rare_mask = (
            (df_train["urgency_level"].isin(["CRITICAL", "HIGH", "MEDIUM"])) |
            (df_train["hazard_type"] != "NONE")
        )
        sample_pool = df_train[rare_mask].copy()
    else:
        sample_pool = df_train.copy()

    indices = list(sample_pool.index)
    rng.shuffle(indices)
    
    ptr = 0
    attempts = 0
    max_attempts = target_count * 15 # Generous retry budget

    strategies = ["composite", "terminology", "restructure", "context"]

    while len(augmented_rows) < target_count and attempts < max_attempts:
        attempts += 1
        idx = indices[ptr % len(indices)]
        ptr += 1
        
        row = sample_pool.loc[idx]
        orig_text = row["text"]
        raw_ents = row["ner_entities"]
        orig_ents = json.loads(raw_ents) if isinstance(raw_ents, str) else raw_ents
        doc_type = row["document_type"]
        
        # Pick strategy
        strategy = rng.choice(strategies)
        
        # Augment document
        new_text, new_ents, success, method_desc = augment_document_preserving_entities(
            orig_text, orig_ents, doc_type, rng, strategy=strategy
        )
        
        if not success:
            qc_gate._reject(f"method_failure_{method_desc}")
            continue

        # Duplicate registry check against existing generated instances
        is_dup, dup_reason = duplicate_registry.is_duplicate(new_text)
        if is_dup:
            qc_gate._reject(dup_reason)
            continue

        # QC Gate 10-point evaluation
        passed, qc_reason = qc_gate.evaluate(
            candidate_text=new_text,
            candidate_entities=new_ents,
            source_row=row.to_dict(),
            augmentation_method=method_desc,
            train_patient_ids=train_patients
        )
        
        if not passed:
            continue

        # Passed all checks! Register text
        duplicate_registry.register(new_text)

        # Build augmented row dictionary preserving all clinical metadata and provenance
        aug_id = f"AUG-{row['document_id']}-{len(augmented_rows)+1:05d}"
        aug_row = {
            "document_id": aug_id,
            "patient_id": row["patient_id"],       # Original patient lineage
            "encounter_id": row["encounter_id"],   # Original encounter lineage
            "document_type": row["document_type"],
            "document_date": row["document_date"],
            "index_date": row["index_date"],
            "text": new_text,
            "cleaned_text": new_text,
            "word_count": len(new_text.split()),
            "char_count": len(new_text),
            "urgency_level": row["urgency_level"],
            "hazard_type": row["hazard_type"],
            "ner_entities": json.dumps(new_ents),
            "slm_summary": row.get("slm_summary", ""),
            "source": f"AUGMENTED_{row.get('source', 'EHR')}",
            "data_split": "TRAIN",
            "quality_status": "VALIDATED_AUGMENTED",
            "disclaimer": row.get("disclaimer", ""),
            "is_augmented": True,
            "source_document_id": row["document_id"],
            "augmentation_method": method_desc
        }
        augmented_rows.append(aug_row)

    print(f"Generated {len(augmented_rows)} / {target_count} candidates in {attempts} attempts.")
    return augmented_rows


def run_pipeline():
    """Main execution function."""
    cfg = load_yaml_config()
    repo_root = CUR_DIR.parent.parent.parent.parent
    train_path = repo_root / cfg["paths"]["train_parquet"]
    val_path = repo_root / cfg["paths"]["validation_parquet"]
    out_dir = repo_root / cfg["paths"]["augmented_data_dir"]
    results_dir = repo_root / cfg["paths"]["results_dir"]
    reports_dir = repo_root / cfg["paths"]["reports_dir"]

    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Loading official TRAIN and VALIDATION partitions...")
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)

    print(f"Loaded {len(df_train)} train documents ({df_train['patient_id'].nunique()} patients).")
    print(f"Loaded {len(df_val)} validation documents ({df_val['patient_id'].nunique()} patients).")

    rng = random.Random(cfg["project"]["random_seed"])
    qc_gate = QualityControlGate(
        min_words=cfg["constraints"]["min_word_count"],
        max_words=cfg["constraints"]["max_word_count"],
        min_chars=cfg["constraints"]["min_char_count"],
        max_chars=cfg["constraints"]["max_char_count"],
        max_jaccard=cfg["constraints"]["jaccard_max_similarity"],
        min_jaccard=cfg["constraints"]["jaccard_min_similarity"]
    )

    # Pre-populate duplicate registry with original train texts
    registry = DuplicateRegistry()
    for t in df_train["text"]:
        registry.register(t)

    # 1. Config A: Save original train with lineage columns
    df_orig = df_train.copy()
    df_orig["is_augmented"] = False
    df_orig["source_document_id"] = df_orig["document_id"]
    df_orig["augmentation_method"] = "none_original"
    df_orig.to_parquet(out_dir / "train_original.parquet", index=False)
    df_orig.to_json(out_dir / "train_original.jsonl", orient="records", lines=True)

    diversity_results = {
        "train_original": compute_dataset_diversity(df_orig)
    }

    # 2. Config B: Original + 25% (1,065 new)
    n_25 = int(round(len(df_train) * 0.25))
    print(f"\n--- Generating Config B (+25% = {n_25} docs) ---")
    aug_25 = generate_augmented_candidates(df_train, n_25, "standard", qc_gate, registry, rng)
    df_aug_25 = pd.concat([df_orig, pd.DataFrame(aug_25)], ignore_index=True)
    df_aug_25.to_parquet(out_dir / "train_augmented_25.parquet", index=False)
    df_aug_25.to_json(out_dir / "train_augmented_25.jsonl", orient="records", lines=True)
    diversity_results["train_augmented_25"] = compute_dataset_diversity(df_aug_25)

    # 3. Config C: Original + 50% (total 2,130 new)
    n_add_c = int(round(len(df_train) * 0.50)) - len(aug_25)
    print(f"\n--- Generating Config C (+50% = total {len(aug_25) + n_add_c} docs) ---")
    aug_add_c = generate_augmented_candidates(df_train, n_add_c, "standard", qc_gate, registry, rng)
    aug_50 = aug_25 + aug_add_c
    df_aug_50 = pd.concat([df_orig, pd.DataFrame(aug_50)], ignore_index=True)
    df_aug_50.to_parquet(out_dir / "train_augmented_50.parquet", index=False)
    df_aug_50.to_json(out_dir / "train_augmented_50.jsonl", orient="records", lines=True)
    diversity_results["train_augmented_50"] = compute_dataset_diversity(df_aug_50)

    # 4. Config D: Original + 100% (total 4,261 new)
    n_add_d = len(df_train) - len(aug_50)
    print(f"\n--- Generating Config D (+100% = total {len(aug_50) + n_add_d} docs) ---")
    aug_add_d = generate_augmented_candidates(df_train, n_add_d, "standard", qc_gate, registry, rng)
    aug_100 = aug_50 + aug_add_d
    df_aug_100 = pd.concat([df_orig, pd.DataFrame(aug_100)], ignore_index=True)
    df_aug_100.to_parquet(out_dir / "train_augmented_100.parquet", index=False)
    df_aug_100.to_json(out_dir / "train_augmented_100.jsonl", orient="records", lines=True)
    diversity_results["train_augmented_100"] = compute_dataset_diversity(df_aug_100)

    # 5. Config E: Targeted Class-Balanced Augmentation
    print(f"\n--- Generating Config E (Targeted Class-Balanced ~1,500 docs) ---")
    targeted_registry = DuplicateRegistry()
    for t in df_train["text"]:
        targeted_registry.register(t)
    aug_targeted = generate_augmented_candidates(df_train, 1500, "targeted", qc_gate, targeted_registry, rng)
    df_aug_targeted = pd.concat([df_orig, pd.DataFrame(aug_targeted)], ignore_index=True)
    df_aug_targeted.to_parquet(out_dir / "train_augmented_targeted.parquet", index=False)
    df_aug_targeted.to_json(out_dir / "train_augmented_targeted.jsonl", orient="records", lines=True)
    diversity_results["train_augmented_targeted"] = compute_dataset_diversity(df_aug_targeted)

    # Leakage Audits for every dataset
    leakage_reports = {}
    for name, df_cand in [
        ("train_augmented_25", df_aug_25),
        ("train_augmented_50", df_aug_50),
        ("train_augmented_100", df_aug_100),
        ("train_augmented_targeted", df_aug_targeted)
    ]:
        aug_only = df_cand[df_cand["is_augmented"] == True]
        is_free, metrics = verify_patient_split_isolation(aug_only, df_val, df_train)
        leakage_reports[name] = metrics
        print(f"Leakage check {name}: {'PASSED (0 leakage)' if is_free else 'FAILED'}")

    # Write metrics to results/
    with open(results_dir / "diversity_metrics.json", "w", encoding="utf-8") as f:
        json.dump(diversity_results, f, indent=2)
    with open(results_dir / "qc_summary.json", "w", encoding="utf-8") as f:
        json.dump(qc_gate.get_summary(), f, indent=2)
    with open(results_dir / "leakage_audit.json", "w", encoding="utf-8") as f:
        json.dump(leakage_reports, f, indent=2)

    print("\nData Augmentation Pipeline execution completed successfully!")


if __name__ == "__main__":
    run_pipeline()
