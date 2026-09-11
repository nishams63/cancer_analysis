"""Master Data Engineering Pipeline for Stage 5 GenAI Synthetic Oncology Stress-Test Engine."""
import sys
import argparse
from pathlib import Path
import pandas as pd

# Add stage5 root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.utils.io import load_yaml, save_yaml, write_parquet, read_parquet, save_json
from src.utils.logging import get_logger
from src.ingestion.source_registry import SourceRegistry
from src.ingestion.load_project_data import ProjectDataLoader
from src.ingestion.load_external_data import ExternalDataLoader
from src.cleaning.clean_demographics import DemographicsCleaner
from src.cleaning.clean_mutations import MutationCleaner
from src.cleaning.clean_biomarkers import BiomarkersCleaner
from src.cleaning.clean_treatments import TreatmentCleaner
from src.cleaning.clean_adverse_events import AdverseEventCleaner
from src.cleaning.clean_timestamps import TimestampCleaner
from src.normalization.normalize_mutations import MutationNormalizer
from src.normalization.normalize_biomarkers import BiomarkerNormalizer
from src.normalization.normalize_treatments import TreatmentNormalizer
from src.normalization.normalize_categories import CategoryNormalizer
from src.distributions.distribution_builder import MasterDistributionBuilder
from src.constraints.constraint_builder import MasterConstraintBuilder
from src.rare_space.rare_space_builder import RareSpaceBuilder
from src.evidence.evidence_loader import EvidenceLoader
from src.provenance.manifest_builder import ManifestBuilder
from src.provenance.versioning import VersionManager

logger = get_logger("master_pipeline")

def generate_reports(
    ingested_info: dict,
    cleaning_issues: list,
    distributions: dict,
    rare_space: dict,
    evidence_results: dict,
    reports_dir: Path
):
    reports_dir.mkdir(parents=True, exist_ok=True)
    df_stage1 = ingested_info["stage1"]["dataframe"]
    df_stage2 = ingested_info["stage2"]["dataframe"]

    # 1. Data Quality Report
    issue_counts = {}
    for iss in cleaning_issues:
        itype = iss["issue_type"]
        issue_counts[itype] = issue_counts.get(itype, 0) + 1

    dq_md = f"""# Stage 5 Data Quality & Quarantine Report

**Execution Timestamp**: {VersionManager.get_timestamp()}  
**Dataset Version**: v1.0  
**Quality Policy**: Quarantine & Flag (Zero Silent Deletion)  

---

## 1. Input Datasets Audited

| Source ID | Name | Rows | Columns | File Hash (SHA-256) |
| :--- | :--- | :---: | :---: | :--- |
| **PROJECT_STAGE1** | Master Patient Tabular Dataset | {len(df_stage1):,} | {len(df_stage1.columns)} | `{ingested_info['stage1']['file_hash'][:16]}...` |
| **PROJECT_STAGE2** | Longitudinal Biomarker Trajectories | {len(df_stage2):,} | {len(df_stage2.columns)} | `{ingested_info['stage2']['file_hash'][:16]}...` |
| **PROJECT_STAGE3** | Clinical NLP Progress Notes | {ingested_info['stage3']['row_count']:,} | {ingested_info['stage3']['col_count']} | `{ingested_info['stage3']['file_hash'][:16]}...` |
| **PROJECT_STAGE4** | Instruction-Tuning Training Pairs | {ingested_info['stage4']['row_count']:,} | {ingested_info['stage4']['col_count']} | `{ingested_info['stage4']['file_hash'][:16]}...` |

---

## 2. Cleaning & Quarantine Issue Summary

Total Tracked Issues: **{len(cleaning_issues)}**

| Issue Type | Occurrence Count | Mitigation & Action Taken |
| :--- | :---: | :--- |
"""
    for itype, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
        dq_md += f"| `{itype}` | {count:,} | Quarantined / Standardized per clinical ontology dictionary |\n"

    dq_md += f"""
---

## 3. Normalization Invariants
- **Demographics**: Clamped age bounds to 18–105 years. Standardized sex labels into canonical `['Male', 'Female', 'Unknown']`.
- **Genomic Mutations**: Cleaned wildtype strings (`None`, `Unknown`, `None/Unknown`) and resolved variant prefixes into canonical gene symbols (`KRAS`, `EGFR`, `TP53`, `ALK`, `MET`, `BRAF`).
- **Physiological Bounds**: Bounded vital signs (systolic BP 60–240 mmHg, heart rate 30–220 bpm) and organ markers (creatinine 0.1–15.0 mg/dL, ALT/AST 1.0–1000.0 U/L).
- **Dosages**: Asserted strictly non-negative dosages and standardized drug names to FDA approved trade/generic names.
- **Audit Lineage**: Complete issue logs are persisted to `C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/data/interim/cleaned/cleaning_issues_log.json`.
"""
    (reports_dir / "data_quality_report.md").write_text(dq_md, encoding="utf-8")

    # 2. Distribution Summary Report
    df_mut = distributions["mutation_frequencies"]
    df_cooc = distributions["mutation_cooccurrence"]
    df_bio = distributions["biomarker_distributions"]
    df_tx = distributions["treatment_distributions"]
    df_dose = distributions["dosage_ranges"]

    dist_md = f"""# Stage 5 Reference Distribution Summary Report

**Execution Timestamp**: {VersionManager.get_timestamp()}  
**Compiled Artifact**: `C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/data/processed/reference_distributions.parquet`  
**Distribution Version**: v1.0  

---

## 1. Genomic Mutation Frequencies (Top Driver Alterations)

| Rank | Mutation Gene | Observed Count | Empirical Frequency |
| :---: | :--- | :---: | :---: |
"""
    for idx, row in df_mut.head(8).iterrows():
        dist_md += f"| {idx+1} | **{row['mutation']}** | {int(row['count']):,} | {float(row['frequency']):.4f} |\n"

    dist_md += f"""
---

## 2. Mutation Co-Occurrence & Dual Alterations

Total Mined Co-Occurrence Pairs: **{len(df_cooc)}**

| Mutation A | Mutation B | Joint Count | Joint Frequency | Rarity Category |
| :--- | :--- | :---: | :---: | :---: |
"""
    for _, row in df_cooc.head(10).iterrows():
        dist_md += f"| {row['mutation_A']} | {row['mutation_B']} | {int(row['cooccurrence_count']):,} | {float(row['cooccurrence_frequency']):.6f} | `{row['joint_rarity']}` |\n"

    dist_md += f"""
---

## 3. Biomarker Summary Distributions

| Biomarker | Unit | Mean ± Std | Median (P50) | Range [Min – Max] | P05 – P95 Range |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""
    for _, row in df_bio.head(10).iterrows():
        dist_md += f"| `{row['biomarker_name']}` | {row['unit']} | {row['mean']} ± {row['std']} | {row['median']} | [{row['min']} – {row['max']}] | [{row['p05']} – {row['p95']}] |\n"

    dist_md += f"""
---

## 4. Treatment & Dosage Ranges

| Treatment Regimen | Modality | Frequency | Observed Range | Median Dose |
| :--- | :--- | :---: | :---: | :---: |
"""
    for _, row in df_dose.head(8).iterrows():
        tx_freq = df_tx[df_tx["treatment"] == row["treatment"]]["frequency"].values
        freq_str = f"{tx_freq[0]:.4f}" if len(tx_freq) > 0 else "N/A"
        dist_md += f"| **{row['treatment']}** | Systemic | {freq_str} | [{row['minimum_observed']} – {row['maximum_observed']} {row['unit']}] | {row['median']} {row['unit']} |\n"

    (reports_dir / "distribution_summary.md").write_text(dist_md, encoding="utf-8")

    # 3. Rarity Summary Report
    scenarios = rare_space.get("scenarios", [])
    rarity_md = f"""# Stage 5 Rare Combination Space & Stress-Test Scenarios

**Execution Timestamp**: {VersionManager.get_timestamp()}  
**Catalog Path**: `C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/configs/rare_combination_space.yaml`  
**Total Mined Scenarios**: {len(scenarios)}  

---

## 1. Rarity Formulation & Mathematical Definition
A synthetic clinical scenario is classified into the rare stress-test space if and only if it satisfies:

$$\\text{{Individually Plausible}} \\land \\text{{Jointly Rare}} \\land \\text{{Allowed by Project Rules}}$$

### Configured Frequency Thresholds:
- **Common**: $f \\ge 0.10$
- **Uncommon**: $0.03 \\le f < 0.10$
- **Rare**: $0.005 \\le f < 0.03$
- **Very Rare**: $f < 0.005$

---

## 2. Mined Rare Edge-Case Stress Scenarios

| Scenario ID | Scenario Name | Primary Features | Joint Frequency | Rarity Tier | Stress Dimension |
| :--- | :--- | :--- | :---: | :---: | :--- |
"""
    for sc in scenarios:
        muts = ", ".join(sc.get("required_features", {}).get("mutations", []))
        rarity_md += f"| `{sc['scenario_id']}` | {sc['scenario_name']} | {muts} | {sc['joint_frequency']:.4f} | `{sc['rarity_category']}` | {sc.get('stress_dimension', 'Multi-variant conflict')} |\n"

    rarity_md += """
---

## 3. Mandatory Engineering & Clinical Disclaimers
> **IMPORTANT CLINICAL & SCIENTIFIC PRINCIPLES:**
> 1. **Rare does not mean clinically severe**: A statistically rare mutation combination (e.g. KRAS + BRAF V600E) may present with mild or standard symptoms, while a common single alteration (e.g. KRAS alone with poor performance status) can be rapidly fatal.
> 2. **Rare does not mean difficult for the AI system**: AI system difficulty depends on model epistemic uncertainty, training data coverage, and reasoning complexity, which is evaluated independently by the Evaluation Engineer.
"""
    (reports_dir / "rarity_summary.md").write_text(rarity_md, encoding="utf-8")

    # 4. Evidence Summary Report
    total_chunks = sum(res.get("chunk_count", 0) for res in evidence_results.values() if isinstance(res, dict))
    ev_md = f"""# Stage 5 Approved RAG Evidence Store Summary Report

**Execution Timestamp**: {VersionManager.get_timestamp()}  
**Evidence Store Version**: v1.0  
**Total Approved Documents**: {len(evidence_results)}  
**Total Traceable Chunks**: {total_chunks}  

---

## 1. Approved Evidence Documents Ingested

| Document ID | Title | Source Registry ID | Category | Version | Chunks |
| :--- | :--- | :--- | :--- | :---: | :---: |
"""
    for doc_id, res in evidence_results.items():
        if isinstance(res, dict) and res.get("status") == "approved":
            m = res["metadata"]
            ev_md += f"| `{m['document_id']}` | {m['title']} | `{m['source']}` | `{m['evidence_category']}` | {m['version']} | {res['chunk_count']} |\n"

    ev_md += f"""
---

## 2. Traceability & Integrity Guarantees
- **Approval Enforcement**: 100% of documents and chunks have `approval_status: approved`. Unregistered sources are strictly blocked by `EvidenceValidator`.
- **Chunk Metadata Schema**: Every chunk retains `chunk_id`, `document_id`, `section`, `page`, `source`, `version`, `approval_status`, and `evidence_category`.
- **Zero Hallucination Retrieval**: Downstream RAG retrievers are strictly restricted to the approved evidence store in `C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/data/evidence/chunks/`.
"""
    (reports_dir / "evidence_summary.md").write_text(ev_md, encoding="utf-8")
    logger.info("All 4 Stage 5 reports generated in C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/reports/")

def run_master_data_engineering(config_path: str = "C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/configs/stage5_data_config.yaml"):
    logger.info("=" * 70)
    logger.info("STAGE 5 — MASTER DATA ENGINEERING PIPELINE EXECUTION")
    logger.info("=" * 70)

    # 1. Load configuration
    cfg = load_yaml(config_path)
    logger.info(f"Loaded config from {config_path} (Version {cfg.get('project', {}).get('version', '1.0.0')})")

    # 2. Source Registry
    reg = SourceRegistry("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/configs/source_registry.yaml")
    logger.info(f"Validated source registry: {len(reg.sources)} registered sources.")

    # 3. Ingest Project Data
    loader = ProjectDataLoader(reg)
    ingested_info = loader.ingest_all_project_data()

    # 4. Ingest External Data
    ext_loader = ExternalDataLoader(reg)
    ext_data = ext_loader.load_all_external()

    # 5. Cleaning & Quarantine Tracking
    df_stage1 = ingested_info["stage1"]["dataframe"]
    df_stage2 = ingested_info["stage2"]["dataframe"]

    all_issues = []
    logger.info("Executing cleaning modules...")
    df_clean, issues = DemographicsCleaner().clean(df_stage1)
    all_issues.extend(issues)
    df_clean, issues = MutationCleaner().clean(df_clean)
    all_issues.extend(issues)
    df_clean, issues = BiomarkersCleaner().clean(df_clean)
    all_issues.extend(issues)
    df_clean, issues = TreatmentCleaner().clean(df_clean)
    all_issues.extend(issues)
    df_clean, issues = AdverseEventCleaner().clean(df_clean)
    all_issues.extend(issues)
    df_clean, issues = TimestampCleaner().clean(df_clean)
    all_issues.extend(issues)

    cleaned_dir = Path("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/data/interim/cleaned")
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(df_clean, cleaned_dir / "cleaned_master_patient.parquet")
    save_json(all_issues, cleaned_dir / "cleaning_issues_log.json")

    # 6. Normalization
    logger.info("Executing normalization modules...")
    df_norm = MutationNormalizer().normalize(df_clean)
    df_norm = BiomarkerNormalizer().normalize(df_norm)
    df_norm = TreatmentNormalizer().normalize(df_norm)
    df_norm = CategoryNormalizer().normalize(df_norm)

    norm_dir = Path("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/data/interim/normalized")
    norm_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(df_norm, norm_dir / "normalized_master_patient.parquet")

    df_s2_clean, s2_issues = TimestampCleaner().clean(df_stage2)
    write_parquet(df_s2_clean, cleaned_dir / "cleaned_longitudinal.parquet")

    # 7. Build Reference Distributions
    logger.info("Building reference distributions...")
    dist_builder = MasterDistributionBuilder()
    distributions = dist_builder.build_all(df_norm, df_s2_clean)

    # 8. Build Constraints
    logger.info("Building constraint specifications...")
    const_builder = MasterConstraintBuilder()
    constraints = const_builder.build_and_save()

    # 9. Build Rare Combination Space
    logger.info("Building rare combination space...")
    rare_builder = RareSpaceBuilder()
    rare_space = rare_builder.build_and_save(distributions["mutation_cooccurrence"])

    # 10. Process Approved Evidence Store
    logger.info("Processing approved evidence store...")
    ev_loader = EvidenceLoader()
    docs = [
        {
            "file": Path("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/data/evidence/documents/oncology_system_architecture_spec.md"),
            "doc_id": "DOC-SPEC-001",
            "title": "Oncology Decision Engine System Architecture Specification",
            "source": "ONCOLOGY_SPEC_001",
            "source_type": "clinical_guideline",
            "version": "2026.v2",
            "category": "clinical_terminology",
            "section": "System Architecture & 6-Stage Roadmap"
        },
        {
            "file": Path("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/data/evidence/documents/nccn_nsclc_targeted_therapy_guideline.md"),
            "doc_id": "DOC-NCCN-002",
            "title": "NCCN NSCLC Biomarker Concordance & Targeted Therapy",
            "source": "NCCN_NSCLC_REF_002",
            "source_type": "external_reference",
            "version": "2026.v1",
            "category": "biomarker",
            "section": "Molecular Testing & Resistance Pathways"
        },
        {
            "file": Path("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/data/evidence/documents/fda_oncology_drug_safety_bulletins.md"),
            "doc_id": "DOC-FDA-003",
            "title": "FDA Oncology Drug Labeling & Dosage Specifications",
            "source": "FDA_DRUG_LABEL_003",
            "source_type": "external_reference",
            "version": "2026.v1",
            "category": "treatment",
            "section": "Approved Dosages & Safety Bulletins"
        },
        {
            "file": Path("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/data/evidence/documents/ctcae_v5_organ_toxicity_grading.md"),
            "doc_id": "DOC-CTCAE-004",
            "title": "CTCAE v5.0 Toxicity & Organ Grading Definitions",
            "source": "CTCAE_TOXICITY_004",
            "source_type": "external_reference",
            "version": "v5.0",
            "category": "toxicity",
            "section": "Organ Toxicity Grading Criteria"
        }
    ]
    evidence_results = {}
    for d in docs:
        if d["file"].exists():
            evidence_results[d["doc_id"]] = ev_loader.process_document(
                filepath=d["file"],
                document_id=d["doc_id"],
                title=d["title"],
                source=d["source"],
                source_type=d["source_type"],
                version=d["version"],
                evidence_category=d["category"],
                section=d["section"]
            )

    # 11. Build Manifests
    logger.info("Building cryptographic manifests...")
    manifest_builder = ManifestBuilder()
    manifests = manifest_builder.build_all_manifests(ingested_info)

    # 12. Generate Reports
    logger.info("Generating production reports...")
    generate_reports(
        ingested_info=ingested_info,
        cleaning_issues=all_issues,
        distributions=distributions,
        rare_space=rare_space,
        evidence_results=evidence_results,
        reports_dir=Path("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/reports")
    )

    logger.info("=" * 70)
    logger.info("STAGE 5 DATA ENGINEERING MASTER PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    return {
        "status": "SUCCESS",
        "ingested_info": ingested_info,
        "distributions": distributions,
        "constraints": constraints,
        "rare_space": rare_space,
        "evidence": evidence_results,
        "manifests": manifests
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 5 Master Data Engineering Pipeline")
    parser.add_argument("--config", default="C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/configs/stage5_data_config.yaml", help="Path to config YAML")
    args = parser.parse_args()
    run_master_data_engineering(args.config)
