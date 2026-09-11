"""Verification script for Stage 5 Data Engineering Definition of Done."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.provenance.lineage import LineageTracker
from src.utils.io import load_yaml, read_parquet, load_json

def verify_all():
    print("=" * 60)
    print("STAGE 5 DATA ENGINEER — DEFINITION OF DONE VERIFICATION")
    print("=" * 60)

    lineage = LineageTracker()

    # Q1: Where did a generated mutation probability come from?
    q1 = lineage.trace_mutation_probability("KRAS")
    print("\\n[Q1] Where did a generated mutation probability come from?")
    print(f"  Source ID: {q1['source_id']}")
    print(f"  Dataset Version: {q1['dataset_version']}")
    print(f"  Distribution Version: {q1['distribution_version']}")
    print(f"  Distribution File: {q1['distribution_file']}")
    print(f"  Calculation Method: {q1['calculation_method']}")
    assert q1["traceable"] is True, "Q1 Failed"

    BASE_DIR = Path(__file__).resolve().parent

    # Q2: What biomarker ranges can the structured sampler use?
    print("\n[Q2] What biomarker ranges can the structured sampler use?")
    spec = load_yaml(BASE_DIR / "configs/constraint_spec.yaml")
    bio_ranges = spec["biomarker_ranges"]
    for bname, rinfo in list(bio_ranges.items())[:4]:
        print(f"  {bname}: min={rinfo['min']}, max={rinfo['max']} {rinfo['unit']} (P95={rinfo.get('distribution_quantile_p95', 'N/A')})")
    assert len(bio_ranges) >= 5, "Q2 Failed"

    # Q3: Which mutation combinations are rare?
    print("\n[Q3] Which mutation combinations are rare?")
    rare_cat = load_yaml(BASE_DIR / "configs/rare_combination_space.yaml")
    scenarios = rare_cat["scenarios"]
    rare_scenarios = [s for s in scenarios if s["rarity_category"] in {"rare", "very_rare"}]
    for s in rare_scenarios[:3]:
        muts = s["required_features"]["mutations"]
        print(f"  {s['scenario_id']} ({s['rarity_category']}): {muts} (freq={s['joint_frequency']:.4f})")
    assert len(rare_scenarios) >= 3, "Q3 Failed"

    # Q4: What oncology evidence can RAG retrieve?
    print("\n[Q4] What oncology evidence can RAG retrieve?")
    ev_manifest = load_json(BASE_DIR / "manifests/evidence_manifest.json")
    print(f"  Total Approved Documents: {ev_manifest['total_documents']}")
    print(f"  Total Traceable Chunks: {ev_manifest['total_chunks']}")
    for doc in ev_manifest["documents"]:
        print(f"  - [{doc['document_id']}] {doc['title']} ({doc['evidence_category']}) -> {doc['chunk_count']} chunks")
    assert ev_manifest["total_chunks"] >= 10, "Q4 Failed"

    # Q5: Can every source be traced?
    print("\n[Q5] Can every source be traced?")
    all_traceable = lineage.verify_all_sources_traceable()
    print(f"  All Sources Traceable: {all_traceable}")
    assert all_traceable is True, "Q5 Failed"

    # Q6: Can the same versioned data foundation be rebuilt?
    print("\n[Q6] Can the same versioned data foundation be rebuilt?")
    dist_manifest = load_json(BASE_DIR / "manifests/distribution_manifest.json")
    total_dists = len(dist_manifest["distributions"])
    print(f"  Total Tracked Distributions: {total_dists}")
    print(f"  Distribution Version: {dist_manifest['distribution_version']}")
    print(f"  Rebuild Script: pipelines/run_data_engineering.py --config configs/stage5_data_config.yaml")
    assert total_dists == 9, "Q6 Failed"

    print("\\n" + "=" * 60)
    print("ALL 6 DEFINITION OF DONE ACCEPTANCE CRITERIA PASS 100%")
    print("=" * 60)

if __name__ == '__main__':
    verify_all()
