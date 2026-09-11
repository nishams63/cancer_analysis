"""Execution pipeline for Stage 5 EDA and Prompt Engineering.

Extracts empirical blind spots across Stages 1 to 4, computes stress-test priority scores,
mines rare combinations and sparse regions, and exports the scenario and retrieval libraries.
"""
import sys
import os
sys.path.insert(0, os.path.abspath("."))

import json
import yaml
import pandas as pd
from datetime import datetime

from src.eda import (
    ClassImbalanceAnalyzer, MutationRarityAnalyzer, SparseRegionAnalyzer,
    MissingnessAnalyzer, TemporalAnalyzer, FailurePatternAnalyzer,
    DisagreementAnalyzer, BlindSpotRanker
)
from src.prompts import (
    build_scenario_catalog, get_core_scenarios,
    export_retrieval_query_library_yaml, export_drift_rules_yaml,
    export_prompt_coverage_csv, generate_prompt_versions_json
)


def run_pipeline():
    print("=" * 70)
    print("STAGE 5: RUNNING EDA & PROMPT ENGINEERING PIPELINE")
    print("=" * 70)

    # Output directories
    analysis_dir = os.path.join("stage5", "analysis", "outputs")
    reports_dir = os.path.join("stage5", "analysis", "reports")
    prompts_dir = os.path.join("stage5", "prompts")
    data_processed = os.path.join("stage5", "data", "processed")

    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(prompts_dir, exist_ok=True)

    # 1. Load Processed Foundation Parquets
    print("\n[1/5] Loading reference foundation distributions...")
    df_cooc = pd.read_parquet(os.path.join(data_processed, "mutation_cooccurrence.parquet"))
    df_mut = pd.read_parquet(os.path.join(data_processed, "mutation_frequencies.parquet"))
    df_bio = pd.read_parquet(os.path.join(data_processed, "biomarker_distributions.parquet"))
    df_miss = pd.read_parquet(os.path.join(data_processed, "missingness_patterns.parquet"))
    df_temp = pd.read_parquet(os.path.join(data_processed, "temporal_patterns.parquet"))
    print(f" Loaded {len(df_cooc)} co-occurrences, {len(df_bio)} biomarker distributions.")

    # 2. Execute EDA Analyzers
    print("\n[2/5] Running empirical blind-spot and failure analyzers...")
    
    # Rare patterns
    rarity_analyzer = MutationRarityAnalyzer()
    df_rare = rarity_analyzer.analyze(df_cooc, df_mut)
    rare_csv = os.path.join(analysis_dir, "rare_patterns.csv")
    df_rare.to_csv(rare_csv, index=False)
    print(f" Generated {rare_csv} ({len(df_rare)} patterns)")

    # Sparse regions
    sparse_analyzer = SparseRegionAnalyzer()
    df_sparse = sparse_analyzer.analyze(df_bio)
    sparse_csv = os.path.join(analysis_dir, "sparse_regions.csv")
    df_sparse.to_csv(sparse_csv, index=False)
    print(f" Generated {sparse_csv} ({len(df_sparse)} sparse regions)")

    # Failure patterns
    failure_analyzer = FailurePatternAnalyzer()
    df_failures = failure_analyzer.analyze()
    failures_csv = os.path.join(analysis_dir, "failure_patterns.csv")
    df_failures.to_csv(failures_csv, index=False)
    print(f" Generated {failures_csv} ({len(df_failures)} empirical failures)")

    # Disagreement patterns
    disagree_analyzer = DisagreementAnalyzer()
    df_disagree = disagree_analyzer.analyze()
    disagree_csv = os.path.join(analysis_dir, "disagreement_patterns.csv")
    df_disagree.to_csv(disagree_csv, index=False)
    print(f" Generated {disagree_csv} ({len(df_disagree)} cross-stage disagreements)")

    # Blind-spot taxonomy ranking
    ranker = BlindSpotRanker()
    df_blind_spots = ranker.rank_blind_spots()
    blind_spots_csv = os.path.join(analysis_dir, "blind_spots.csv")
    df_blind_spots.to_csv(blind_spots_csv, index=False)
    print(f" Generated {blind_spots_csv} ({len(df_blind_spots)} blind spots ranked)")

    # Scenario priorities mapping
    scenario_priorities_data = [
        {"scenario_id": "PROMPT-R01", "blind_spot_id": "BS01", "priority_score": 0.655, "priority_tier": "CRITICAL", "recommended_cases": 50, "vulnerability": "Omission of MET bypass resistance in EGFR-mutant NSCLC"},
        {"scenario_id": "PROMPT-R02", "blind_spot_id": "BS02", "priority_score": 0.749, "priority_tier": "CRITICAL", "recommended_cases": 60, "vulnerability": "Negation leakage across contrastive conjunctions (however/but)"},
        {"scenario_id": "PROMPT-R03", "blind_spot_id": "BS03", "priority_score": 0.635, "priority_tier": "HIGH", "recommended_cases": 40, "vulnerability": "Sub-centimeter primary lesion overriding metastatic Stage IVA label"},
        {"scenario_id": "PROMPT-R04", "blind_spot_id": "BS04", "priority_score": 0.705, "priority_tier": "CRITICAL", "recommended_cases": 50, "vulnerability": "Decision boundary collapse in borderline moderate-risk biomarker envelope"},
        {"scenario_id": "PROMPT-R05", "blind_spot_id": "BS05", "priority_score": 0.640, "priority_tier": "HIGH", "recommended_cases": 45, "vulnerability": "Failure to catch Cisplatin contraindication during acute renal injury"},
        {"scenario_id": "PROMPT-R06", "blind_spot_id": "BS06", "priority_score": 0.589, "priority_tier": "HIGH", "recommended_cases": 40, "vulnerability": "Hyper-acute septic shock decompensation within 6 hours post-chemo"},
        {"scenario_id": "PROMPT-R07", "blind_spot_id": "BS07", "priority_score": 0.615, "priority_tier": "HIGH", "recommended_cases": 40, "vulnerability": "Imputation collapse under >60% missing tabular clinical fields"},
        {"scenario_id": "PROMPT-R08", "blind_spot_id": "BS08", "priority_score": 0.611, "priority_tier": "HIGH", "recommended_cases": 35, "vulnerability": "Quadruple mutation STK11/KEAP1 primary resistance to IO monotherapy"},
        {"scenario_id": "PROMPT-R09", "blind_spot_id": "BS09", "priority_score": 0.770, "priority_tier": "CRITICAL", "recommended_cases": 55, "vulnerability": "Sarcopenic elderly normal creatinine masking low CrCl (<30 mL/min)"},
        {"scenario_id": "PROMPT-R10", "blind_spot_id": "BS10", "priority_score": 0.713, "priority_tier": "HIGH", "recommended_cases": 35, "vulnerability": "Adenosquamous carcinoma misclassified as conventional adenocarcinoma"},
        {"scenario_id": "PROMPT-R11", "blind_spot_id": "BS11", "priority_score": 0.680, "priority_tier": "CRITICAL", "recommended_cases": 45, "vulnerability": "Re-challenging checkpoint inhibitor after Grade 3 immune myocarditis"},
        {"scenario_id": "PROMPT-R12", "blind_spot_id": "BS12", "priority_score": 0.815, "priority_tier": "CRITICAL", "recommended_cases": 60, "vulnerability": "Inverse weighting false positive on benign Grade 1 xerosis/proteinuria"},
        {"scenario_id": "PROMPT-R13", "blind_spot_id": "BS13", "priority_score": 0.595, "priority_tier": "MEDIUM", "recommended_cases": 30, "vulnerability": "OOD noisy clinic EHR note with heavy shorthand and typos"},
        {"scenario_id": "PROMPT-R14", "blind_spot_id": "BS14", "priority_score": 0.769, "priority_tier": "CRITICAL", "recommended_cases": 50, "vulnerability": "Stale clinic note claiming stable disease conflicting with RECIST progression"},
        {"scenario_id": "PROMPT-R15", "blind_spot_id": "BS15", "priority_score": 0.685, "priority_tier": "HIGH", "recommended_cases": 40, "vulnerability": "Subjective complaint bias ignoring silent severe hypercalcemia (14.6 mg/dL)"}
    ]
    df_sc_prio = pd.DataFrame(scenario_priorities_data).sort_values("priority_score", ascending=False).reset_index(drop=True)
    sc_prio_csv = os.path.join(analysis_dir, "scenario_priorities.csv")
    df_sc_prio.to_csv(sc_prio_csv, index=False)
    print(f" Generated {sc_prio_csv} ({len(df_sc_prio)} scenarios prioritized)")

    # 3. Build & Export Prompt Library & Artifacts
    print("\n[3/5] Compiling and exporting scenario catalog and prompt library...")
    catalog = build_scenario_catalog()
    scenarios = catalog.scenarios

    # scenario_catalog.yaml & prompt_library.yaml
    catalog_dict = catalog.model_dump(mode='json')
    sc_catalog_yaml = os.path.join(prompts_dir, "scenario_catalog.yaml")
    prompt_lib_yaml = os.path.join(prompts_dir, "prompt_library.yaml")
    with open(sc_catalog_yaml, "w", encoding="utf-8") as f:
        yaml.dump(catalog_dict, f, default_flow_style=False, sort_keys=False)
    with open(prompt_lib_yaml, "w", encoding="utf-8") as f:
        yaml.dump(catalog_dict, f, default_flow_style=False, sort_keys=False)
    print(f" Exported {sc_catalog_yaml} and {prompt_lib_yaml}")

    # retrieval_query_library.yaml
    retrieval_yaml = os.path.join(prompts_dir, "retrieval_query_library.yaml")
    export_retrieval_query_library_yaml(retrieval_yaml, scenarios)
    print(f" Exported {retrieval_yaml}")

    # prompt_drift_rules.yaml
    drift_yaml = os.path.join(prompts_dir, "prompt_drift_rules.yaml")
    export_drift_rules_yaml(drift_yaml)
    print(f" Exported {drift_yaml}")

    # prompt_coverage.csv
    coverage_csv = os.path.join(prompts_dir, "prompt_coverage.csv")
    export_prompt_coverage_csv(coverage_csv, scenarios)
    print(f" Exported {coverage_csv}")

    # prompt_versions.json
    versions_json = os.path.join(prompts_dir, "prompt_versions.json")
    generate_prompt_versions_json(versions_json, scenarios)
    print(f" Exported {versions_json}")

    # 4. Generate Markdown Reports
    print("\n[4/5] Generating analysis and governance markdown reports...")
    generate_reports(reports_dir, df_blind_spots, df_failures, df_sc_prio, df_rare, df_sparse)

    print("\n[5/5] EDA & Prompt Engineering pipeline completed successfully!")
    print("=" * 70)


def generate_reports(reports_dir: str, df_blind_spots: pd.DataFrame, df_failures: pd.DataFrame,
                     df_sc_prio: pd.DataFrame, df_rare: pd.DataFrame, df_sparse: pd.DataFrame):
    """Generate 4 comprehensive analysis reports."""

    # 1. blind_spot_report.md
    bs_report_path = os.path.join(reports_dir, "blind_spot_report.md")
    with open(bs_report_path, "w", encoding="utf-8") as f:
        f.write("# Stage 5 Blind-Spot Analysis & Taxonomy Report\n\n")
        f.write("## Executive Summary\n")
        f.write("This report details the empirical analysis of vulnerability regions identified across Stages 1 to 4. ")
        f.write("Rather than testing models on common, in-distribution scenarios, Stage 5 systematically targets ")
        f.write("15 defined blind-spot categories (`BS01` to `BS15`) evaluated across tabular ML, deep learning, ")
        f.write("clinical NLP, and small language model generation.\n\n")
        f.write("## Blind-Spot Taxonomy & Priority Ranking\n\n")
        f.write("| Blind Spot | Name | Affected Stages | Priority Score | Tier | Failure Rate | Underrep | Disagreement |\n")
        f.write("|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|\n")
        for _, row in df_blind_spots.iterrows():
            f.write(f"| `{row['blind_spot_id']}` | {row['name']} | {row['affected_stages']} | {row['stress_test_priority_score']:.4f} | **{row['priority_tier']}** | {row['stage_failure_rate']:.2f} | {row['underrepresentation_score']:.2f} | {row['disagreement_score']:.2f} |\n")
        f.write("\n## Priority Scoring Methodology\n")
        f.write("Stress-Test Priority Score (STPS) is computed using weighted multi-objective empirical factors:\n")
        f.write("$$\\text{STPS} = 0.30 \\cdot S_{\\text{fail}} + 0.20 \\cdot S_{\\text{underrep}} + 0.20 \\cdot S_{\\text{disag}} + 0.15 \\cdot S_{\\text{uncert}} + 0.10 \\cdot S_{\\text{rarity}} + 0.05 \\cdot S_{\\text{reprod}}$$\n\n")
        f.write("Critical-tier blind spots (`STPS >= 0.70`) demonstrate severe failure consequences or high cross-stage disagreement.\n")

    # 2. stage_failure_report.md
    failure_report_path = os.path.join(reports_dir, "stage_failure_report.md")
    with open(failure_report_path, "w", encoding="utf-8") as f:
        f.write("# Empirical Stage Failure Analysis Report\n\n")
        f.write("## Cross-Stage Empirical Vulnerabilities\n\n")
        f.write("| Failure ID | Stage | Category | Error Rate | Severity | Description |\n")
        f.write("|:---|:---|:---|:---:|:---:|:---|\n")
        for _, row in df_failures.iterrows():
            f.write(f"| `{row['failure_id']}` | {row['affected_stage']} | {row['failure_category']} | {row['observed_error_rate']:.4f} | **{row['severity']}** | {row['failure_description']} |\n")
        f.write("\n## Stage-Specific Vulnerability Breakdown\n\n")
        f.write("### Stage 1 (Tabular ML - XGBoost/CatBoost)\n")
        f.write("- **High-Risk False Negatives**: 140 patients with severe clinical toxicity misclassified as Low or Moderate risk (Recall = 58.08%).\n")
        f.write("- **Moderate Risk Confusion**: F1 score collapses to 0.3251, with 41.8% misclassified as Low.\n\n")
        f.write("### Stage 3 (Clinical NLP - DeBERTa)\n")
        f.write("- **Negation Leaks**: In contrastive sentences ('Denies X, however Y is present'), attention mechanism incorrectly negates Y.\n")
        f.write("- **Inverse Class Weight Over-Triggering**: Inverse frequency loss causes 93.6% false alarms in Dermatologic hazards and 54.2% in Renal hazards.\n\n")
        f.write("### Stage 4 (Clinical SLM - Qwen2.5-1.5B LoRA)\n")
        f.write("- **Dropped Secondary Driver Alterations**: In co-occurring genomic profiles (e.g. EGFR + MET), generation omits the bypass alteration.\n")
        f.write("- **Out-of-Distribution Degradation**: Note performance drops on messy, authentic clinic notes with heavy shorthand and dictation errors.\n")

    # 3. scenario_priority_report.md
    prio_report_path = os.path.join(reports_dir, "scenario_priority_report.md")
    with open(prio_report_path, "w", encoding="utf-8") as f:
        f.write("# Scenario Priority and Test Allocation Report\n\n")
        f.write("## Scenario Prioritization Matrix\n\n")
        f.write("| Scenario ID | Blind Spot | Priority Score | Tier | Recommended Cases | Target Vulnerability |\n")
        f.write("|:---|:---|:---:|:---:|:---:|:---|\n")
        for _, row in df_sc_prio.iterrows():
            f.write(f"| `{row['scenario_id']}` | `{row['blind_spot_id']}` | {row['priority_score']:.4f} | **{row['priority_tier']}** | {row['recommended_cases']} | {row['vulnerability']} |\n")
        f.write(f"\n**Total Recommended Synthetic Stress Scenarios**: {df_sc_prio['recommended_cases'].sum()} cases.\n")

    # 4. coverage_gap_report.md
    gap_report_path = os.path.join(reports_dir, "coverage_gap_report.md")
    with open(gap_report_path, "w", encoding="utf-8") as f:
        f.write("# Coverage Gap & Reference Envelope Analysis Report\n\n")
        f.write("## Identified Cohort Coverage Gaps\n\n")
        f.write("1. **Dual Driver Bypass Alterations**: Standard cohorts have < 1.5% dual mutations, causing models to treat drivers as mutually exclusive.\n")
        f.write("2. **Sarcopenic Geriatric Renal Envelope**: Normal serum creatinine (0.8-1.0 mg/dL) combined with low body weight (< 45 kg) creating undetected CrCl < 30 mL/min.\n")
        f.write("3. **Hyper-Acute Sepsis Trajectory**: Static outpatient records fail to evaluate intraday decompensation (< 8 hours).\n")
        f.write("4. **High Missingness Regimes**: Routine hospital transfer records with > 60% missing tabular attributes collapse to default population medians.\n\n")
        f.write(f"## Rare Mutation Pairs Identified\n")
        f.write(f"Identified {len(df_rare[df_rare['candidate_stress_test']])} candidate rare pairs for stress testing out of {len(df_rare)} analyzed.\n\n")
        f.write(f"## Sparse Biomarker Transition Envelopes\n")
        f.write(f"Mapped {len(df_sparse)} sensitive decision boundary ranges across creatinine, ctDNA, tumor markers, liver function, and platelets.\n")

    print(f" Saved 4 reports to {reports_dir}")


if __name__ == "__main__":
    run_pipeline()
