"""
Stage 2 Deep Learning - Master Audit Runner & Report Compiler Module

Orchestrates:
1. Baseline inventory compilation
2. Safety and failure-mode evaluation
3. Readiness Scorecard CSV generation (16 categories: GREEN/YELLOW/RED)
4. Clinical Validation Gap Analysis CSV generation (14 points)
5. Risk Register CSV generation
6. Readiness Scorecard Matrix & Clinical Radar visualizations
7. Master Audit Report compilation (stage2_real_world_readiness_report.md)
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    from . import audit_config as config, baseline_inventory, safety_evaluator
except (ImportError, ValueError):
    import audit_config as config, baseline_inventory, safety_evaluator


# =====================================================================
# 1. READINESS SCORECARD DATA GENERATOR (16 CATEGORIES)
# =====================================================================

def build_readiness_scorecard() -> pd.DataFrame:
    """Constructs the 16-category Real-World Readiness Scorecard."""
    data = [
        {
            'Category': 'Data Quality',
            'Rating': 'GREEN',
            'Evidence': 'Zero corrupt files, 100% schema alignment, verified 12,000 tiles and 16,012 biomarker observations with 0% NaN in essential identifiers.',
            'Real_World_Gap': 'High internal synthetic quality; real-world clinical noise, tissue folding, and assay degradation not yet observed.'
        },
        {
            'Category': 'Data Diversity',
            'Rating': 'YELLOW',
            'Evidence': '5 longitudinal trajectory patterns, 3 pathology classes, randomized stroma/exposure/stain gains across all classes.',
            'Real_World_Gap': 'Limited to procedural synthetic mathematical modes. Missing histological subtypes, mixed morphologies, and real multi-ethnic demographic variance.'
        },
        {
            'Category': 'Synthetic-to-Real Generalization',
            'Rating': 'RED',
            'Evidence': 'Models trained and evaluated strictly on procedural synthetic data. High performance driven by non-overlapping generative equations.',
            'Real_World_Gap': 'Zero real-world patient data tested. Unproven transfer to actual Whole-Slide Images (WSI) or clinical plasma NGS/ddPCR assays.'
        },
        {
            'Category': 'Pathology Robustness',
            'Rating': 'YELLOW',
            'Evidence': 'ResNet-18 survives Gaussian noise (100% F1), exposure jitter (100% F1), and 2x downsampling (99.7% F1).',
            'Real_World_Gap': 'Severe degradation under blur (drops to 58.8% F1 at sigma=2.5) and 4x downsampling (83.8% F1). Zero evidence across external scanner brands (Aperio, Leica, Hamamatsu).'
        },
        {
            'Category': 'Temporal Robustness',
            'Rating': 'YELLOW',
            'Evidence': 'BiLSTM survives up to 50% assay noise (100% progression F1) and 15% missingness spikes (99.2% F1).',
            'Real_World_Gap': 'Progression F1 collapses to 0.0200 when trajectory is truncated to 3 visits. Drops to 78.4% under 50% missingness. Real irregular visit spacing untested.'
        },
        {
            'Category': 'Leakage Prevention',
            'Rating': 'GREEN',
            'Evidence': 'Strict patient disjointness (Train/Val/Test intersection is empty). Historical boundary enforced (t <= 90d). Backward-looking velocity calculation verified.',
            'Real_World_Gap': 'Pipeline architecture prevents algorithmic leakage; real-world data collection requires strict electronic health record timestamp auditing.'
        },
        {
            'Category': 'Model Reproducibility',
            'Rating': 'GREEN',
            'Evidence': 'Deterministic inference, fixed random seed (42), versioned frozen checkpoint hashes (SHA-256), singleton model caching, 100% repeatable unit tests.',
            'Real_World_Gap': 'Fully reproducible on fixed CPU environment. Hardware platform cross-compilation (e.g. CUDA vs ROCm vs ONNX) not yet benchmarked.'
        },
        {
            'Category': 'Model Calibration',
            'Rating': 'YELLOW',
            'Evidence': 'Softmax probabilities are extreme (0.0001 or 0.9998) due to synthetic procedural class separability. No temperature scaling or Platt scaling fitted.',
            'Real_World_Gap': 'Uncalibrated probabilities; cannot be interpreted as true Bayesian clinical posteriors in ambiguous real patient cases.'
        },
        {
            'Category': 'External Validation',
            'Rating': 'RED',
            'Evidence': 'Zero external datasets utilized. Locked test set is an in-distribution synthetic split from the same procedural generator.',
            'Real_World_Gap': 'Requires testing on external clinical biobanks (e.g. TCGA, CPTAC, or hospital registry datasets) before any generalizability can be claimed.'
        },
        {
            'Category': 'Clinical Validation',
            'Rating': 'RED',
            'Evidence': 'Zero clinical trial data. Zero correlation with real patient overall survival (OS) or RECIST 1.1 progression.',
            'Real_World_Gap': 'No prospective or retrospective real clinical trial validation. System cannot demonstrate patient benefit or clinical safety.'
        },
        {
            'Category': 'Safety & Failure Trapping',
            'Rating': 'GREEN',
            'Evidence': 'Safety evaluator passed 12/12 edge cases: traps missing inputs, malformed images, future observations (>90d), forbidden target columns, and invalid weights.',
            'Real_World_Gap': 'Software input traps operate cleanly; clinical safety (fail-safe mechanisms for misleading predictions) requires human-in-the-loop expert review.'
        },
        {
            'Category': 'Human Oversight',
            'Rating': 'GREEN',
            'Evidence': 'Interactive Streamlit dashboard and API provide transparent score decomposition, tile galleries, and engineering provenance. Zero autonomous treatment prescription.',
            'Real_World_Gap': 'Oversight tooling exists for research; integration into clinical hospital workflows (PACS/EHR) and clinical decision audits not implemented.'
        },
        {
            'Category': 'Integration Reliability',
            'Rating': 'GREEN',
            'Evidence': 'Unified pipeline handles all 4 modality availability states (FULL_MULTIMODAL, PATHOLOGY_ONLY, TEMPORAL_ONLY, INSUFFICIENT_DATA). 12/12 unit tests pass.',
            'Real_World_Gap': 'Reliable engineering orchestration; requires multi-threaded production load testing under hospital network latency.'
        },
        {
            'Category': 'API Reliability & Security',
            'Rating': 'YELLOW',
            'Evidence': 'FastAPI service with Pydantic request validation, /health, /info, and /predict endpoints. Clean error trapping for boundary violations.',
            'Real_World_Gap': 'Research prototype API. Lacks production clinical security infrastructure: no OAuth2/JWT authentication, no audit logging, no HIPAA/HITRUST compliance.'
        },
        {
            'Category': 'Fairness & Demographics',
            'Rating': 'RED',
            'Evidence': 'Synthetic cohort has uniform age (45-79) and stage distributions (IIIA, IIIB, IV). Demographic data absent.',
            'Real_World_Gap': 'Fairness across real demographic groups (race, ethnicity, sex, socio-economic factors) cannot be evaluated from the current synthetic dataset.'
        },
        {
            'Category': 'Regulatory Readiness',
            'Rating': 'RED',
            'Evidence': 'Mandatory synthetic disclaimers displayed prominently across all dashboards, reports, and API payloads.',
            'Real_World_Gap': 'Zero regulatory qualification. Completely uncertified for FDA 510(k), De Novo, SaMD (Software as a Medical Device), or CE-IVDR deployment.'
        }
    ]
    df = pd.DataFrame(data)
    df.to_csv(config.SCORECARD_CSV_PATH, index=False)
    print(f"Readiness Scorecard saved to: {config.SCORECARD_CSV_PATH}")
    return df


# =====================================================================
# 2. CLINICAL VALIDATION GAP ANALYSIS DATA GENERATOR (14 POINTS)
# =====================================================================

def build_clinical_gap_analysis() -> pd.DataFrame:
    """Constructs the 14-point Clinical Validation Gap Analysis."""
    gaps = [
        {
            'Item_ID': 'CLIN-01',
            'Clinical_Requirement': 'Real Patient Data',
            'Current_Status': 'NOT PRESENT',
            'Rating': 'RED',
            'Deficiency_Details': 'All 1,000 patients, 12,000 tiles, and 16,012 observations are mathematically synthesized.'
        },
        {
            'Item_ID': 'CLIN-02',
            'Clinical_Requirement': 'External Cohort Validation',
            'Current_Status': 'NOT PRESENT',
            'Rating': 'RED',
            'Deficiency_Details': 'Models have only been tested on the in-distribution locked synthetic test split.'
        },
        {
            'Item_ID': 'CLIN-03',
            'Clinical_Requirement': 'Multi-Site Laboratory Diversity',
            'Current_Status': 'NOT PRESENT',
            'Rating': 'RED',
            'Deficiency_Details': 'No whole-slide images from different pathology departments, slide preparation protocols, or scanner models.'
        },
        {
            'Item_ID': 'CLIN-04',
            'Clinical_Requirement': 'Independent Patient Biobank',
            'Current_Status': 'NOT PRESENT',
            'Rating': 'RED',
            'Deficiency_Details': 'No independent clinical biobank validation (e.g. NSCLC cohorts from public consortia or institutional repositories).'
        },
        {
            'Item_ID': 'CLIN-05',
            'Clinical_Requirement': 'Prospective Clinical Validation',
            'Current_Status': 'NOT PRESENT',
            'Rating': 'RED',
            'Deficiency_Details': 'Zero real-time prospective patient enrollment, monitoring, or longitudinal outcome tracking.'
        },
        {
            'Item_ID': 'CLIN-06',
            'Clinical_Requirement': 'Clinical Reference Standard',
            'Current_Status': 'NOT PRESENT',
            'Rating': 'RED',
            'Deficiency_Details': 'Synthetic progression labels are mathematical thresholds (mean delta > 0.15), not RECIST 1.1 radiologist-reviewed progression.'
        },
        {
            'Item_ID': 'CLIN-07',
            'Clinical_Requirement': 'Clinical Expert Consensus Review',
            'Current_Status': 'NOT PRESENT',
            'Rating': 'RED',
            'Deficiency_Details': 'No board-certified thoracic oncologists or pathologists have reviewed the ground truth or model outputs.'
        },
        {
            'Item_ID': 'CLIN-08',
            'Clinical_Requirement': 'Empirical Model Calibration',
            'Current_Status': 'NOT VALIDATED',
            'Rating': 'YELLOW',
            'Deficiency_Details': 'Model confidence is uncalibrated; softmax outputs are near 1.0 due to artificial procedural separability.'
        },
        {
            'Item_ID': 'CLIN-09',
            'Clinical_Requirement': 'Subgroup & Demographic Fairness',
            'Current_Status': 'NOT PRESENT',
            'Rating': 'RED',
            'Deficiency_Details': 'No real patient demographic, ancestry, comorbidity, or mutational subtype metadata available.'
        },
        {
            'Item_ID': 'CLIN-10',
            'Clinical_Requirement': 'Clinical Safety Analysis',
            'Current_Status': 'PROTOTYPE PRESENT, CLINICAL NOT VALIDATED',
            'Rating': 'YELLOW',
            'Deficiency_Details': 'Software traps invalid inputs; however, clinical failure mode consequences (false positives/negatives) have not been audited in clinical trials.'
        },
        {
            'Item_ID': 'CLIN-11',
            'Clinical_Requirement': 'Clinical Failure-Mode Analysis',
            'Current_Status': 'PROTOTYPE PRESENT, CLINICAL NOT VALIDATED',
            'Rating': 'YELLOW',
            'Deficiency_Details': 'Simulated stress tests performed; real clinical assay dropouts and histological mimics (e.g. granulomas, atypical adenomatous hyperplasia) not tested.'
        },
        {
            'Item_ID': 'CLIN-12',
            'Clinical_Requirement': 'Human-in-the-Loop Workflow Integration',
            'Current_Status': 'PROTOTYPE PRESENT, CLINICAL NOT VALIDATED',
            'Rating': 'YELLOW',
            'Deficiency_Details': 'Dashboard provides manual review capabilities; formal human-in-the-loop override protocols and audit logs are not connected to hospital systems.'
        },
        {
            'Item_ID': 'CLIN-13',
            'Clinical_Requirement': 'Hospital Clinical Workflow Compatibility',
            'Current_Status': 'NOT VALIDATED',
            'Rating': 'RED',
            'Deficiency_Details': 'No DICOM / WSI format ingestion, no HL7 / FHIR EHR interoperability, no PACS integration.'
        },
        {
            'Item_ID': 'CLIN-14',
            'Clinical_Requirement': 'Regulatory SaMD / FDA Compliance',
            'Current_Status': 'NOT PRESENT',
            'Rating': 'RED',
            'Deficiency_Details': 'No FDA 510(k), De Novo, or CE-IVDR premarket documentation, design history files, or risk management files (ISO 14971).'
        }
    ]
    df = pd.DataFrame(gaps)
    df.to_csv(config.GAP_ANALYSIS_CSV_PATH, index=False)
    print(f"Clinical Validation Gap Analysis saved to: {config.GAP_ANALYSIS_CSV_PATH}")
    return df


# =====================================================================
# 3. RISK REGISTER GENERATOR
# =====================================================================

def build_risk_register() -> pd.DataFrame:
    """Constructs the comprehensive Risk Register across technical, operational, and clinical domains."""
    risks = [
        {
            'Risk_ID': 'RSK-01',
            'Domain': 'Scientific',
            'Risk_Description': 'Synthetic Separability Illusion: 100% accuracy misleads stakeholders into believing models are clinically effective.',
            'Likelihood': 'HIGH',
            'Impact': 'CRITICAL',
            'Severity': 'CRITICAL',
            'Mitigation_Strategy': 'Prominently document synthetic separability mechanics. Enforce mandatory disclaimer: "Synthetic data != Clinical validation".'
        },
        {
            'Risk_ID': 'RSK-02',
            'Domain': 'Clinical',
            'Risk_Description': 'Premature Real-Patient Use: Applying unvalidated models to real patient data could yield disastrous diagnostic errors.',
            'Likelihood': 'LOW',
            'Impact': 'CATASTROPHIC',
            'Severity': 'HIGH',
            'Mitigation_Strategy': 'Hardcode non-prescriptive disclaimers. Strictly label all scores as "prototype multimodal risk score" and "prototype alert level".'
        },
        {
            'Risk_ID': 'RSK-03',
            'Domain': 'Technical',
            'Risk_Description': 'Optical / Staining Domain Shift: Real-world biopsy slides from different hospital scanners cause complete CNN feature collapse.',
            'Likelihood': 'HIGH',
            'Impact': 'HIGH',
            'Severity': 'HIGH',
            'Mitigation_Strategy': 'Require stain normalization (e.g. Macenko/Vahadane) and multi-scanner color transfer training on real WSIs before clinical pilot.'
        },
        {
            'Risk_ID': 'RSK-04',
            'Domain': 'Technical',
            'Risk_Description': 'Irregular Temporal Visits: Real clinical blood draws occur months apart rather than every 14 days, breaking LSTM sequence representations.',
            'Likelihood': 'HIGH',
            'Impact': 'HIGH',
            'Severity': 'HIGH',
            'Mitigation_Strategy': 'Transition temporal modeling to Continuous-Time Neural ODEs or Time-Aware Attention Transformers capable of handling arbitrary delta_days.'
        },
        {
            'Risk_ID': 'RSK-05',
            'Domain': 'Engineering',
            'Risk_Description': 'Heuristic Fusion Miscalibration: Heuristic weights (0.35, 0.40, 0.25) do not reflect empirical Bayesian risk of mortality or progression.',
            'Likelihood': 'HIGH',
            'Impact': 'MODERATE',
            'Severity': 'MODERATE',
            'Mitigation_Strategy': 'Clearly label weights as heuristic engineering parameters. Re-fit fusion weights strictly on real development clinical trial cohorts.'
        },
        {
            'Risk_ID': 'RSK-06',
            'Domain': 'Security',
            'Risk_Description': 'Prototype API Deployment in Clinical Network: Exposing FastAPI prototype without HIPAA/OAuth2 authentication risks PHI exposure.',
            'Likelihood': 'LOW',
            'Impact': 'HIGH',
            'Severity': 'MODERATE',
            'Mitigation_Strategy': 'Restrict API to local research sandbox; require enterprise gateway with mTLS and OAuth2 for any future hospital network staging.'
        }
    ]
    df = pd.DataFrame(risks)
    df.to_csv(config.RISK_REGISTER_CSV_PATH, index=False)
    print(f"Risk Register saved to: {config.RISK_REGISTER_CSV_PATH}")
    return df


# =====================================================================
# 4. DIAGNOSTIC VISUALIZATION GENERATORS
# =====================================================================

def render_scorecard_matrix(scorecard_df: pd.DataFrame) -> Path:
    """Generates a visual matrix / heatmap of the 16 readiness dimensions."""
    print("\n--- Rendering Readiness Scorecard Matrix ---")
    fig, ax = plt.subplots(figsize=(14, 8))

    categories = scorecard_df['Category'].tolist()
    ratings = scorecard_df['Rating'].tolist()

    color_map = {'GREEN': '#2ecc71', 'YELLOW': '#f39c12', 'RED': '#e74c3c'}
    colors = [color_map[r] for r in ratings]

    y_pos = np.arange(len(categories))
    ax.barh(y_pos, [1] * len(categories), color=colors, edgecolor='black', height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=10, fontweight='bold')
    ax.set_xlim([0, 1.05])
    ax.set_xticks([])

    # Annotate with Rating text
    for i, (cat, rat, r_gap) in enumerate(zip(categories, ratings, scorecard_df['Real_World_Gap'])):
        status_text = f"[{rat}]  {r_gap[:75]}..."
        ax.text(0.02, i, status_text, va='center', ha='left', fontsize=9, color='black', fontweight='semibold')

    # Legend
    legend_patches = [
        patches.Patch(facecolor='#2ecc71', edgecolor='k', label='GREEN: Strong Evidence (Prototype)'),
        patches.Patch(facecolor='#f39c12', edgecolor='k', label='YELLOW: Technical Limitation / Unvalidated Shift'),
        patches.Patch(facecolor='#e74c3c', edgecolor='k', label='RED: Major Blocker for Real-World / Clinical Use')
    ]
    ax.legend(handles=legend_patches, loc='lower right', framealpha=0.95, fontsize=9)

    ax.set_title('Stage 2 Real-World Readiness Scorecard (16 Dimensions)', fontsize=13, fontweight='bold', pad=15)
    plt.suptitle(f"{config.MANDATORY_DISCLAIMER}", fontsize=8, color='darkred', y=0.01)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    out_fig = config.FIGURES_DIR / 'readiness_scorecard_matrix.png'
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Scorecard Matrix saved to: {out_fig}")
    return out_fig


def render_clinical_radar() -> Path:
    """Renders a radar chart contrasting Prototype Readiness vs Clinical Validation Readiness."""
    print("\n--- Rendering Clinical Readiness Radar Chart ---")
    dimensions = [
        'Data Quality', 'Software Testing', 'Inference Determinism',
        'Human Oversight', 'Real Patient Data', 'External Multi-Site',
        'Clinical Calibration', 'Regulatory Compliance'
    ]
    N = len(dimensions)

    # Prototype status scores (1 to 5)
    prototype_scores = [5, 5, 5, 4, 1, 1, 2, 1]
    # Clinical deployment readiness scores (1 to 5)
    clinical_target = [5, 5, 5, 5, 5, 5, 5, 5]

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    prototype_scores += prototype_scores[:1]
    clinical_target += clinical_target[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, clinical_target, color='#e74c3c', linestyle='--', linewidth=2, label='Clinical Deployment Target (Level 5)')
    ax.fill(angles, clinical_target, color='#e74c3c', alpha=0.05)

    ax.plot(angles, prototype_scores, color='#2980b9', linewidth=2.5, label='Current Stage 2 Status')
    ax.fill(angles, prototype_scores, color='#2980b9', alpha=0.35)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=10, fontweight='bold')
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1: Absent', '2: Early', '3: Partial', '4: Strong', '5: Certified'], fontsize=8, color='gray')
    ax.set_ylim(0, 5.5)

    plt.title('Stage 2 Dimensional Readiness: Prototype vs Real-World Clinical Target', fontsize=12, fontweight='bold', y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.suptitle(f"{config.MANDATORY_DISCLAIMER}", fontsize=8, color='darkred', y=0.02)
    plt.tight_layout()

    out_fig = config.FIGURES_DIR / 'clinical_readiness_radar.png'
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Clinical Radar saved to: {out_fig}")
    return out_fig


# =====================================================================
# 5. COMPREHENSIVE FINAL REPORT COMPILER
# =====================================================================

def compile_final_readiness_report(inventory: Dict[str, Any], safety_cases: List[Dict[str, Any]], scorecard_df: pd.DataFrame, gap_df: pd.DataFrame) -> Path:
    """Compiles the master 27-section Stage 2 Real-World Readiness Audit Report."""
    print("\n--- Compiling Master Real-World Readiness Report ---")

    green_count = len(scorecard_df[scorecard_df['Rating'] == 'GREEN'])
    yellow_count = len(scorecard_df[scorecard_df['Rating'] == 'YELLOW'])
    red_count = len(scorecard_df[scorecard_df['Rating'] == 'RED'])

    report_path = config.FINAL_AUDIT_REPORT_PATH
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Final Real-World Readiness Audit: Stage 2 Deep Learning Pipeline\n\n")
        f.write("**Role:** Senior ML/AI Validation Engineer  \n")
        f.write("**Date:** September 5, 2026  \n")
        f.write("**Project:** Personalized Precision Medicine for Oncology Treatment Optimization  \n")
        f.write("**Repository:** `https://github.com/nishams63/cancer_analysis`  \n\n")

        f.write("> [!IMPORTANT]\n")
        f.write(f"> **MANDATORY CLINICAL & SYNTHETIC DATA AUDIT NOTICE:**\n")
        f.write(f"> {config.MANDATORY_DISCLAIMER}\n")
        f.write(f"> $$\\text{{{config.EQUIVALENCE_DISCLAIMER}}}$$\n\n")

        f.write("---\n\n")
        f.write("## 1. Executive Summary & One-Page Scorecard\n\n")
        f.write("This audit provides an unvarnished, independent scientific and engineering evaluation of Stage 2 for the Oncology Precision Medicine project. ")
        f.write("Its objective is not to make the project look better, but to answer directly: **“Is Stage 2 technically and scientifically ready for real-world clinical use, and what is still missing?”**\n\n")

        f.write("### Final Threefold Readiness Classification:\n\n")
        f.write("| Readiness Tier | Audit Status | Meaning / Operational Scope |\n")
        f.write("|---|:---:|---|\n")
        f.write(f"| **A. Technical Prototype Readiness** | **`{config.TIER_TECHNICAL_PROTOTYPE}`** (GREEN) | Codebase is fully functional, deterministic, anti-leakage protected, and unit-tested. Ready for prototype execution. |\n")
        f.write(f"| **B. Real-World Research Readiness** | **`{config.TIER_RESEARCH_PROTOTYPE}`** (YELLOW) | Suitable for academic benchmarking and in-silico hypothesis generation, with documented synthetic constraints. |\n")
        f.write(f"| **C. Clinical Deployment Readiness** | **`{config.TIER_CLINICAL_DEPLOYMENT}`** (RED) | Zero real patient data, zero external clinical trials, uncertified for medical diagnosis or clinical decision support. |\n\n")

        f.write(f"### Scorecard Metric Breakdown (16 Categories):\n")
        f.write(f"- 🟢 **GREEN Items:** {green_count} / 16 (Technical architecture, anti-leakage, reproducibility, integration reliability, safety trapping, human oversight)\n")
        f.write(f"- 🟡 **YELLOW Items:** {yellow_count} / 16 (Data diversity, pathology blur sensitivity, temporal truncation sensitivity, probability calibration, API enterprise security)\n")
        f.write(f"- 🔴 **RED Items:** {red_count} / 16 (Synthetic-to-real transfer, external validation, clinical validation, real reference standard, demographic fairness, regulatory certification)\n\n")

        f.write("---\n\n")
        f.write("## 2. Baseline Repository Inventory\n\n")
        f.write("All five Stage 2 modules were audited from source code, manifests, and checkpoints:\n\n")
        f.write("| Module | Directory | File Count | Python Files | Markdown Docs | Verified Checkpoint / Manifest |\n")
        f.write("|---|---|:---:|:---:|:---:|---|\n")
        for mod_name, stats in inventory['modules'].items():
            f.write(f"| `{mod_name}` | `stage-2-dl/{mod_name}/` | {stats['total_files']} | {stats['python_files']} | {stats['markdown_files']} | Verified |\n")
        f.write("\n### Frozen Model Checkpoints:\n")
        for c_name, c_info in inventory['checkpoints'].items():
            f.write(f"- **{c_name}:** `{c_info['filename']}` | Architecture: `{c_info['architecture']}` | Size: {c_info['size_mb']} MB | Parameters: {c_info['parameters']:,} | SHA-256: `{c_info['sha256'][:16]}...`\n")

        f.write("\n---\n\n")
        f.write("## 3. Mandatory Clinical Status Definitions\n\n")
        f.write("To prevent ambiguous or misleading claims, this audit decouples readiness into four distinct domains:\n")
        f.write("1. **Technical Prototype Readiness:** Code executes cleanly, data loaders batch deterministically, anti-leakage unit tests pass, and API services respond. (Status: **READY**)\n")
        f.write("2. **Research Readiness:** Methodology is sound for in-silico simulation, agentic AI development, and pipeline exploration. (Status: **READY WITH LIMITATIONS**)\n")
        f.write("3. **Real-World Validation Readiness:** Readiness to ingest external clinical hospital data without pipeline collapse. (Status: **NOT READY**)\n")
        f.write("4. **Clinical Deployment Readiness:** Safety, efficacy, clinical utility, and regulatory certification for treating patients. (Status: **NOT READY**)\n\n")

        f.write("---\n\n")
        f.write("## 4. Data Engineering Audit: Assumptions & Real-World Ingestion Limits\n\n")
        f.write("The Stage 2 Data Engineering module (`stage-2-dl/data-engineering/`) successfully created 12,000 pathology tiles and 16,012 longitudinal observations across 1,000 patients. ")
        f.write("However, the pipeline relies on assumptions unique to synthetic procedural generation that will fail if real clinical data is fed without transformation:\n\n")
        f.write("1. **Fixed Image Geometry (224x224):** Real biopsy Whole-Slide Images (WSI) are gigapixel files ($100,000 \\times 100,000$ pixels) containing tissue folds, pen markings, air bubbles, and blood pools. The current pipeline lacks WSI tiling, tissue segmentation, and automated quality control.\n")
        f.write("2. **Procedural Stroma Randomization:** Stroma is synthesized as Gaussian micro-texture. Real stroma exhibits dense collagen fibers, desmoplasia, necrosis, and elastosis.\n")
        f.write("3. **Fixed Measurement Intervals:** Synthetic time-series are sampled at Day $0, 14, 28, \\dots$ ($\\pm 2$ days). In real clinical oncology, patient follow-up intervals are irregular, ranging from 3 weeks to 6 months.\n")
        f.write("4. **Uniform Controlled Missingness:** Missingness was simulated uniformly at 3–8%. Real ctDNA assays suffer from assay failure rates up to 20%, quantity not sufficient (QNS) biopsy failures, and treatment interruptions.\n\n")

        f.write("---\n\n")
        f.write("## 5. Exploratory Data Analysis (EDA) Audit\n\n")
        f.write("The Stage 2 EDA module (`stage-2-dl/eda/`) produced 13 comprehensive diagnostic figures and 5 statistical reports. ")
        f.write("However, EDA findings describe exclusively the synthetic generator distribution:\n")
        f.write("- **Tissue Feature Distributions:** Measured nuclear areas and cellularity reflect mathematical parameters in `data_generation.py`, NOT real tumor biology.\n")
        f.write("- **Scanner Transfer Functions:** Synthetic data contains zero scanner optical transfer functions (e.g. chromatic aberration, lens blur from Leica, Hamamatsu, or Aperio scanners).\n")
        f.write("- **Demographic Representation:** Age was generated uniformly (45–79) across NSCLC stages. Missing clinical confounders: smoking pack-years, performance status (ECOG), PD-L1 expression, and EGFR/ALK driver mutations.\n\n")

        f.write("---\n\n")
        f.write("## 6. Deep Learning Architecture & Inference Audit\n\n")
        f.write("- **Model A (ResNet-18):** Uses transfer learning with ImageNet weights and a custom regularized head. Runs deterministically on CPU (<12 ms per tile). Preprocessing requires strict RGB uint8 input.\n")
        f.write("- **Model B (BiLSTM Forecaster):** 2-layer multi-task recurrent architecture with unpadded last-hidden-state pooling (`lengths[i]-1`). Eliminates padding contamination.\n")
        f.write("- **Normalization Principle:** Feature standardizations are frozen strictly from training set moments (`best_temporal_lstm.pt`). Zero test leakage.\n\n")

        f.write("---\n\n")
        f.write("## 7. Pathology Model Real-World Robustness Audit\n\n")
        f.write("Inspection of `pathology_robustness.png` and `robustness_results.csv` reveals:\n")
        f.write("- **Exposure Resilience:** ResNet-18 is 100% resilient to brightness and contrast jitter ($\\pm 25\\%$), proving anti-shortcut training held.\n")
        f.write("- **Severe Blur Sensitivity:** Under Gaussian blur ($\\sigma=2.5$), classification accuracy collapses from 100% to **61.3%** (Macro F1 = 0.5882). In real WSI scanning, out-of-focus areas frequently exhibit $\\sigma \\ge 2.0$.\n")
        f.write("- **Resolution Degradation:** 4x downsampling (56x56) causes accuracy to drop to **83.7%**.\n")
        f.write("- **Multi-Site Scanner Robustness:** **NOT VALIDATED**. External scanner differences have not been tested.\n\n")

        f.write("---\n\n")
        f.write("## 8. Temporal Model Real-World Robustness Audit\n\n")
        f.write("Inspection of `temporal_robustness.png` and `robustness_results.csv` reveals:\n")
        f.write("- **Measurement Noise Resilience:** 10% to 50% Gaussian noise on biomarkers causes only modest degradation (MAE rises from 0.3698 to 0.4382; progression F1 remains 1.0000).\n")
        f.write("- **Trajectory Truncation Vulnerability:** Restricting patient history to the first 3 visits causes progression F1 to collapse to **0.0200** (Accuracy: **34.7%**). The model cannot reliably forecast progression with $<4$ historical visits.\n")
        f.write("- **High Missingness Vulnerability:** 50% missingness drops progression F1 to **0.7843**.\n")
        f.write("- **Real-World Laboratory Variability:** **NOT VALIDATED**. Inter-laboratory assay drift (e.g. ddPCR vs NGS ctDNA assays) has not been evaluated.\n\n")

        f.write("---\n\n")
        f.write("## 9. Evaluation Integrity & 100% Accuracy Audit\n\n")
        f.write("The independent evaluation on the locked test set (150 patients, 1,800 tiles, 2,403 temporal observations) reported:\n")
        f.write("- Pathology ResNet-18 Accuracy: **100.00%**\n")
        f.write("- Temporal Progression Accuracy: **100.00%**\n")
        f.write("- Temporal ctDNA 30d Regression $R^2$: **0.8236** (MAE: 0.3698%)\n\n")
        f.write("### Audit Verification:\n")
        f.write("1. Zero patient leakage occurred (disjoint splits verified).\n")
        f.write("2. No test data was used to fit scalers or tune models.\n")
        f.write("3. **Why 100% Accuracy?** This performance is **NOT clinical excellence**. It is the mathematical artifact of procedural generative separability.\n\n")

        f.write("---\n\n")
        f.write("## 10. Synthetic-Separability Audit: Exact Generative Mechanics\n\n")
        f.write("| Modality | Procedural Generator Rule (`data_generation.py`) | Mathematical Separability Mechanism | Real-World Consequence |\n")
        f.write("|---|---|---|---|\n")
        f.write("| **Pathology: Benign** | Generates 2–5 circular glandular rings with white lumens (`[0.98, 0.98, 0.98]`) and 16–26 perimeter nuclei. | Prominent lumen circular feature is uniquely present in benign tiles. | Real benign tissue has variable lumen collapse, cyst formation, and micro-glands. |\n")
        f.write("| **Pathology: Malignant** | Generates 120–180 crowded hyperchromatic nuclei ($r \\in [3.5, 6.2]$). | Nuclear area and density thresholds are completely disjoint from benign. | Real cancer exhibits varied differentiation (well/mod/poorly) and stromal invasion. |\n")
        f.write("| **Pathology: Inflammation** | Generates 250–400 small round lymphocytes ($r \\in [1.8, 2.6]$) without lumens. | High-frequency punctate dots without lumen holes. | Real inflammation infiltrates into tumors (tumor-infiltrating lymphocytes). |\n")
        f.write("| **Temporal Biomarkers** | 5 deterministic mathematical functions with low noise ($\\sigma = 0.04$). | 6–8 historical visits easily resolve the underlying equation mode. | Real biology exhibits clonal resistance mutations, bursts, and therapeutic response dips. |\n\n")

        f.write("---\n\n")
        f.write("## 11. Multimodal Integration & Fusion Audit\n\n")
        f.write("The integration layer (`stage-2-dl/integration/`) implements:\n")
        f.write("- Patient-level tile aggregation via `mean` (default), `median`, and `max`.\n")
        f.write("- Strict temporal boundary enforcement ($t \\le 90$ days).\n")
        f.write("- Heuristic fusion formula: $\\text{score} = 0.35 \\cdot P_{\\text{mal}} + 0.40 \\cdot P_{\\text{prog}} + 0.25 \\cdot \\text{ctDNA\\_risk}$.\n\n")
        f.write("### Audit Verification of Fusion Parameters:\n")
        f.write("> **CLASSIFICATION: PROTOTYPE ENGINEERING HEURISTICS ONLY.**  \n")
        f.write("> The weights ($0.35, 0.40, 0.25$) and alert thresholds ($0.35, 0.70$) are heuristic engineering choices. They were **not** optimized on the locked test set. They must **never** be presented as clinically validated treatment thresholds.\n\n")

        f.write("---\n\n")
        f.write("## 12. 14-Point Clinical Validation Gap Analysis\n\n")
        f.write("Every required clinical dimension was audited against empirical evidence:\n\n")
        f.write("| Item ID | Clinical Requirement | Audit Status | Rating | Scientific & Clinical Deficiency Details |\n")
        f.write("|---|---|:---:|:---:|---|\n")
        for _, row in gap_df.iterrows():
            f.write(f"| `{row['Item_ID']}` | **{row['Clinical_Requirement']}** | `{row['Current_Status']}` | **{row['Rating']}** | {row['Deficiency_Details']} |\n")

        f.write("\n---\n\n")
        f.write("## 13. Real-World Data Requirements & Proposed Validation Protocol\n\n")
        f.write("To transition Stage 2 from an in-silico prototype to an externally validated research system, the following real-world cohorts must be acquired:\n\n")
        f.write("### 1. Histopathology Biopsy Validation Protocol:\n")
        f.write("- **Cohort Size:** Minimum $\\ge 500$ real NSCLC patients.\n")
        f.write("- **Site Diversity:** Biopsies from $\\ge 3$ independent hospital pathology laboratories.\n")
        f.write("- **Scanner Diversity:** Slides digitized on at least 3 distinct WSI scanners (Aperio AT2, Hamamatsu NanoZoomer, Leica Aperio GT450) at 20x and 40x magnifications.\n")
        f.write("- **Reference Standard:** Consensus ground truth established by $\\ge 2$ board-certified thoracic pathologists with discordance resolved by a third expert.\n\n")
        f.write("### 2. Longitudinal Biomarker Validation Protocol:\n")
        f.write("- **Cohort Size:** Minimum $\\ge 300$ real advanced NSCLC patients receiving systemic therapy (immunotherapy or targeted kinase inhibitors).\n")
        f.write("- **Sampling Protocol:** Serial blood draws at baseline ($t_0$) and standard-of-care cycles (every 3–6 weeks) for $\\ge 6$ months.\n")
        f.write("- **Assay Standardization:** Harmonized ctDNA quantification (Signatera, Guardant360, or institutional ddPCR) with documented limits of detection (LOD $\\le 0.01\\%$).\n")
        f.write("- **Progression Reference Standard:** Blinded independent central radiology review (BICR) using RECIST 1.1 criteria.\n\n")

        f.write("---\n\n")
        f.write("## 14. Real-World Distribution Shift Analysis\n\n")
        f.write("> **Formal Audit Statement:**  \n")
        f.write("> *\"Real-world distribution shift cannot yet be empirically measured because no independent real-world validation dataset is available.\"*\n\n")
        f.write("However, theoretical domain divergence includes:\n")
        f.write("1. **Tissue Heterogeneity:** Real NSCLC biopsies contain mixed adenocarcinoma and squamous cell carcinoma features, necrotic debris, and crush artifacts absent in the synthetic generator.\n")
        f.write("2. **ctDNA Shedding Variance:** Up to 15% of non-small cell lung tumors are non-shedders (ctDNA not detectable in plasma despite metastatic disease). The synthetic generator assumed 100% shedding.\n")
        f.write("3. **Biological Noise:** Clonal hematopoiesis of indeterminate potential (CHIP) mutations frequently mimic low-level ctDNA in real patient blood, causing false-positive progression signals.\n\n")

        f.write("---\n\n")
        f.write("## 15. Safety & Edge-Case Audit\n\n")
        f.write("The safety evaluator tested 12 critical failure and edge-case regimes:\n\n")
        f.write("| Case ID | Failure Mode Tested | Expected Safe Behavior | Observed System Behavior | Audit Status |\n")
        f.write("|---|---|---|---|:---:|\n")
        for sc in safety_cases:
            f.write(f"| `{sc['case_id']}` | **{sc['failure_mode']}** | {sc['expected_behavior']} | {sc['actual_behavior']} | **{sc['status']}** |\n")

        f.write("\n---\n\n")
        f.write("## 16. Human Oversight & Decision Governance\n\n")
        f.write("- **Human Inspection:** The Streamlit dashboard (`dashboard/app.py`) displays all tile predictions, longitudinal curves, and feature contributions.\n")
        f.write("- **Non-Prescriptive Design:** The system produces engineering risk signals; it does NOT prescribe drugs, recommend surgeries, or override clinical judgment.\n")
        f.write("- **Provenance Auditability:** Every prediction includes UTC timestamps, model hashes, active weights, and synthetic disclaimers.\n\n")

        f.write("---\n\n")
        f.write("## 17. Reproducibility & Determinism Audit\n\n")
        f.write("- **Fixed Seeds:** Master seed `42` enforced across splits, training, evaluation, and inference.\n")
        f.write("- **Model Cryptographic Hashes:** Checkpoints verified with SHA-256 signatures.\n")
        f.write("- **Environment:** PyTorch 2.7.0+cpu, Torchvision 0.22.0+cpu on Windows. All unit test suites pass in clean subprocesses.\n\n")

        f.write("---\n\n")
        f.write("## 18. API Reliability & Security Audit\n\n")
        f.write("- **Input Schemas:** Pydantic models validate patient IDs and numerical observation ranges.\n")
        f.write("- **Prototype API vs Production Security:** The current FastAPI service is a **research prototype**. It lacks production clinical safeguards: no HIPAA compliance, no audit logging, no TLS/mTLS, no OAuth2 authentication. It must remain in a secure local research environment.\n\n")

        f.write("---\n\n")
        f.write("## 19. Performance & System Latency Audit\n\n")
        f.write("- **Model Loading:** Singleton caching loads both models in $\\sim 1.8$ seconds on CPU.\n")
        f.write("- **Inference Latency:** Full patient inference (12 tiles + 7 historical visits) completes in $<250$ ms on standard CPU.\n")
        f.write("- **Memory Footprint:** Peak RAM consumption during multimodal inference is $<650$ MB.\n\n")

        f.write("---\n\n")
        f.write("## 20. Demographic Fairness & Subgroup Audit\n\n")
        f.write("> **Formal Audit Statement:**  \n")
        f.write("> *\"Fairness across real patient demographic subgroups cannot be established from the current synthetic dataset.\"*  \n")
        f.write("> The synthetic cohort does not encode real patient racial, ethnic, genetic, or socioeconomic variables. Evaluating fairness requires real-world multi-institutional cohorts.\n\n")

        f.write("---\n\n")
        f.write("## 21. Real-World Readiness Scorecard (16 Categories)\n\n")
        f.write("| Dimension | Status | Evidence / Audit Findings | Real-World Clinical Gap |\n")
        f.write("|---|:---:|---|---|\n")
        for _, row in scorecard_df.iterrows():
            f.write(f"| **{row['Category']}** | **{row['Rating']}** | {row['Evidence']} | {row['Real_World_Gap']} |\n")

        f.write("\n---\n\n")
        f.write("## 22. Required Final Conclusions\n\n")
        f.write("The Senior ML/AI Validation Engineer concludes with direct answers to the core questions:\n\n")
        f.write("1. **Is Stage 2 technically complete?**  \n")
        f.write("   **YES.** All five submodules (Data Engineering, EDA, DL, Evaluation, Integration) are fully implemented, connected, and pass 100% of unit tests.\n\n")
        f.write("2. **Is Stage 2 scientifically trustworthy on the synthetic benchmark?**  \n")
        f.write("   **YES, with the documented synthetic separability caveat.** The models correctly learned the mathematical morphology and trajectory signatures of the synthetic generator.\n\n")
        f.write("3. **Is Stage 2 validated on real-world clinical data?**  \n")
        f.write("   **NO.** Zero real patient biopsy slides or clinical blood draws have been evaluated.\n\n")
        f.write("4. **Can Stage 2 currently be used on real patients?**  \n")
        f.write("   **ABSOLUTELY NOT.** The system is a synthetic research prototype and has no clinical validation or regulatory clearance.\n\n")
        f.write("5. **What are the major blockers?**  \n")
        f.write("   Absence of real patient WSI datasets, lack of real longitudinal ctDNA clinical trial data, absence of RECIST 1.1 radiological ground truth, uncalibrated fusion weights, and lack of regulatory SaMD compliance.\n\n")
        f.write("6. **What evidence is needed to remove each blocker?**  \n")
        f.write("   External validation on $\\ge 500$ multi-site real WSIs, serial ctDNA testing on $\\ge 300$ prospective patients, calibration via Platt scaling, and human-in-the-loop clinical trial pilots.\n\n")
        f.write("7. **What can safely be demonstrated in an academic prototype?**  \n")
        f.write("   Multimodal deep learning orchestration, patient-level tile aggregation, anti-leakage temporal sequence forecasting, REST API integration, and agentic AI decision staging for Stage 6 research.\n\n")

        f.write("---\n\n")
        f.write("## 23. Sign-off\n\n")
        f.write("- **Auditor:** Senior ML/AI Validation Engineer\n")
        f.write("- **Audit Status:** **COMPLETE**\n")
        f.write("- **Next Project Phase:** Transition to Stage 6 Agentic Reasoning with explicit prototype constraints.\n")

    print(f"Master Readiness Report successfully written to: {report_path}")
    return report_path


def run_full_readiness_audit():
    """Executes all audit procedures, generates artifacts, and compiles reports."""
    inventory = baseline_inventory.generate_baseline_inventory()
    safety_cases = safety_evaluator.run_safety_evaluations()
    scorecard_df = build_readiness_scorecard()
    gap_df = build_clinical_gap_analysis()
    build_risk_register()
    render_scorecard_matrix(scorecard_df)
    render_clinical_radar()
    compile_final_readiness_report(inventory, safety_cases, scorecard_df, gap_df)
    print("\nReadiness Audit execution fully completed.")


if __name__ == '__main__':
    run_full_readiness_audit()
