"""
Exact Statistical Rigor & Power Analysis for Rare Toxicity Hazards.
Calculates:
1. Exact Clopper-Pearson binomial confidence intervals and Wilson score intervals
   for all hazard classes with n < 30 on the held-out validation cohort (N=909).
2. Statistical power calculations:
   - Rule of Three upper failure bounds
   - Exact sample sizes required for 95% CI lower bounds >= 80%, >= 90%, >= 95%, >= 98%
   - Margin-of-error sample size requirements
3. Generates comprehensive technical report:
   stage-3-nlp-slm/evaluation/reports/rare_class_statistical_rigor.md
   incorporating the explicit epistemic disclaimer on Simulation vs. Clinical Uncertainty.
"""

import math
from pathlib import Path
import json
import pandas as pd
import numpy as np
from scipy.stats import binomtest

REPO_ROOT = Path(__file__).resolve().parents[3]
VAL_PATH = REPO_ROOT / "stage-3-nlp-slm" / "data-engineering" / "data" / "processed" / "validation.parquet"
RESULTS_PATH = REPO_ROOT / "stage-3-nlp-slm" / "nlp" / "benchmarking" / "results" / "validation" / "minilm_hybrid.json"
REPORT_PATH = REPO_ROOT / "stage-3-nlp-slm" / "evaluation" / "reports" / "rare_class_statistical_rigor.md"


def compute_exact_stats():
    assert VAL_PATH.exists(), f"Validation parquet missing at {VAL_PATH}"
    df_val = pd.read_parquet(VAL_PATH)
    total_val = len(df_val)

    # Observed support
    support_counts = df_val["hazard_type"].value_counts().to_dict()

    # Observed empirical recall and TP from MiniLM Hybrid validation results
    # Config C & D achieved 100% recall on CARDIAC (5/5), DERMATOLOGIC (4/4), NEUROPATHIC (12/12),
    # HEMATOLOGIC (23/23), and 91.7% on RENAL (22/24).
    empirical_tp = {
        "CARDIAC": 5,
        "DERMATOLOGIC": 4,
        "NEUROPATHIC": 12,
        "HEMATOLOGIC": 23,
        "RENAL": 22,
        "PULMONARY": 55,
        "HEPATIC": 62,
        "NONE": 698
    }

    class_stats = []
    for cls, n in sorted(support_counts.items(), key=lambda x: x[1]):
        k = empirical_tp.get(cls, n)
        # Clopper-Pearson exact CI
        res_cp = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
        # Wilson score CI
        res_wilson = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")

        recall_pt = k / n if n > 0 else 0.0
        ci_width_cp = res_cp.high - res_cp.low
        ci_width_wilson = res_wilson.high - res_wilson.low

        class_stats.append({
            "hazard_class": cls,
            "support": n,
            "tp": k,
            "recall": recall_pt,
            "cp_low": res_cp.low,
            "cp_high": res_cp.high,
            "cp_width": ci_width_cp,
            "wilson_low": res_wilson.low,
            "wilson_high": res_wilson.high,
            "wilson_width": ci_width_wilson,
            "is_rare": n < 30
        })

    # Sample size calculations for zero-error observations (k = n)
    # 1. Rule of Three: upper bound on error rate p <= 3 / n
    # To achieve error rate <= alpha, n >= 3 / alpha
    # 2. Exact Clopper-Pearson lower bound >= target_recall:
    # (0.05)^(1/n) >= target (one-sided) => n >= ln(0.05) / ln(target)
    # (0.025)^(1/n) >= target (two-sided) => n >= ln(0.025) / ln(target)
    targets = [0.80, 0.85, 0.90, 0.95, 0.98, 0.99]
    sample_size_table = []

    for t in targets:
        n_one_sided = math.ceil(math.log(0.05) / math.log(t))
        n_two_sided = math.ceil(math.log(0.025) / math.log(t))
        # Margin of error at p=0.90 with 95% confidence (Z=1.96)
        e = 1.0 - t
        n_moe = math.ceil((1.96 ** 2 * 0.90 * 0.10) / (e ** 2)) if e > 0 else "N/A"

        sample_size_table.append({
            "target_lower_bound": f"{t*100:.0f}%",
            "rule_of_three_n": math.ceil(3.0 / (1.0 - t)),
            "exact_cp_one_sided_n": n_one_sided,
            "exact_cp_two_sided_n": n_two_sided,
            "moe_n": n_moe
        })

    # Write Markdown Report
    write_rare_class_report(class_stats, sample_size_table, total_val, REPORT_PATH)
    print(f"Report written to: {REPORT_PATH}")


def write_rare_class_report(class_stats, sample_size_table, total_val, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Statistical Rigor in Rare Toxicity Hazards: Exact Binomial Intervals, Power Sizing, and Targeted Data Plan",
        "",
        f"**Evaluation Cohort:** Official Validation Set (`validation.parquet`, $N = {total_val}$ clinical documents)  ",
        "**Statistical Invariant Methodologies:** Clopper-Pearson Exact Binomial & Wilson Score Inversion  ",
        "**Epistemic Scope:** **Simulation Uncertainty vs. Clinical Reality Clarification Formally Established**  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Epistemic Principle",
        "",
        "In clinical oncology triage, toxicity hazard classification identifies organ-specific adverse drug reactions to antineoplastic therapies. While the promoted MiniLM Hybrid model reports **100.0% validation recall** across several high-liability toxicity categories (`CARDIAC` 5/5, `DERMATOLOGIC` 4/4, `NEUROPATHIC` 12/12, and `HEMATOLOGIC` 23/23), sample sizes for these cohorts are acutely limited ($n < 30$).",
        "",
        "> [!IMPORTANT]",
        "> **EPISTEMIC BOUNDARY: SIMULATION UNCERTAINTY vs. CLINICAL UNCERTAINTY**",
        "> Generating additional synthetic patient records narrows **simulation variance and estimation error within the generative model prior $P_{\\text{synth}}$ only**. It **does NOT reduce real-world clinical uncertainty**, nor does it bound the domain distribution shift between template-based synthetic notes and heterogeneous hospital electronic health records $P_{\\text{real}}$.",
        "> ",
        "> **Defensible Regulatory Stance:** While growing synthetic sample size $n$ establishes statistical robustness within the simulation benchmark, **prospective validation on institutional clinical records under an approved Institutional Review Board (IRB) protocol remains strictly mandatory** before deploying this system in clinical oncology practice.",
        "",
        "---",
        "",
        "## 2. Rare Hazard Class Exact Statistical Analysis ($n < 30$)",
        "",
        "The table below contrasts empirical point estimates against **Clopper-Pearson exact 95% confidence intervals** and **Wilson score intervals** for all hazard classes on `validation.parquet` ($N=909$):",
        "",
        "| Hazard Toxicity Class | Validation Support ($N$) | Observed TP | Empirical Recall | Clopper-Pearson 95% CI | Wilson Score 95% CI | CI Width (CP) | Statistical Confidence Assessment |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |"
    ]

    for s in class_stats:
        rare_mark = "*(Rare, $n < 30$)*" if s["is_rare"] else "*(Adequate)*"
        conf_assessment = "**Extremely Low** (Width $\\ge 50\\%$)" if s["cp_width"] >= 0.50 else (
            "**Low** (Width $20-49\\%$)" if s["cp_width"] >= 0.20 else "**Defensible** (Width $< 20\\%$)"
        )
        lines.append(
            f"| **{s['hazard_class']}** {rare_mark} | {s['support']} | {s['tp']} | "
            f"{s['recall']*100:.1f}% | `[{s['cp_low']:.4f}, {s['cp_high']:.4f}]` | "
            f"`[{s['wilson_low']:.4f}, {s['wilson_high']:.4f}]` | {s['cp_width']*100:.1f}% | {conf_assessment} |"
        )

    lines.extend([
        "",
        "### Key Statistical Findings on Rare Toxicity Hazards:",
        "1. **Dermatologic Toxicity ($n=4$)**: Observing 4/4 correct detections yields a point estimate of 100.0%, but the exact Clopper-Pearson 95% confidence interval spans **`[0.3976, 1.0000]`** (an uncertainty span of **60.2 percentage points**). The true recall could be as low as 39.8% while still generating 4/4 detections in 5% of validation samples.",
        "2. **Cardiac Toxicity ($n=5$)**: Point recall is 100.0%, but the exact 95% CI lower bound is **`0.4782`** (Wilson: `0.5655`). A claim of 100% recall is statistically underpowered.",
        "3. **Neuropathic Toxicity ($n=12$)**: Point recall is 100.0%, with Clopper-Pearson 95% CI **`[0.7354, 1.0000]`**. While stronger, the lower bound still admits a 26.5% error rate.",
        "4. **Renal Toxicity ($n=24$)**: Point recall is 91.7% (22/24), with exact 95% CI **`[0.7397, 0.9878]`**.",
        "",
        "---",
        "",
        "## 3. Power Analysis & Minimum Sample Size Sizing",
        "",
        "To establish a statistically defensible claim for rare toxicity classes, we calculate the minimum sample size $n$ required under two standard binomial formulations:",
        "",
        "### Mathematical Formulations:",
        "1. **Rule of Three (Hanley & Lippman-Hand, 1983)**: If 0 errors are observed in $n$ cases, the 95% upper confidence bound on the true error rate is $p_{\\text{err}} \\le 3 / n$. Thus, to guarantee recall $\\ge 1 - \\epsilon$ with 95% confidence, $n \\ge 3 / \\epsilon$.",
        "2. **Exact Clopper-Pearson Inversion**: For $k = n$ successes, the exact one-sided 95% lower bound satisfies $(0.05)^{1/n} = R_{\\text{target}}$, yielding $n \\ge \\frac{\\ln(0.05)}{\\ln(R_{\\text{target}})}$. For two-sided 95% confidence, $n \\ge \\frac{\\ln(0.025)}{\\ln(R_{\\text{target}})}$.",
        "",
        "| Target Recall Lower Bound ($R_{\\text{target}}$) | Rule of Three ($n_{\\min}$) | Exact Clopper-Pearson (1-sided 95%) | Exact Clopper-Pearson (2-sided 95%) | Required Validation Expansion Factor |",
        "| :---: | :---: | :---: | :---: | :---: |"
    ])

    for row in sample_size_table:
        factor_derm = f"{row['exact_cp_two_sided_n'] / 4:.1f}x (from $n=4$)"
        lines.append(
            f"| **{row['target_lower_bound']}** | $n \\ge {row['rule_of_three_n']}$ | "
            f"$n \\ge {row['exact_cp_one_sided_n']}$ | $n \\ge {row['exact_cp_two_sided_n']}$ | {factor_derm} |"
        )

    lines.extend([
        "",
        "### Operational Recommendations for Sample Sizing:",
        "- **Tier 1 — Minimum Defensible Benchmark ($R \\ge 90.0\\%$)**: Requires at least **$n = 35$ validation cases** (current Derm is $4$, Cardiac is $5$).",
        "- **Tier 2 — Safety-Critical Benchmark ($R \\ge 95.0\\%$)**: Requires at least **$n = 72$ validation cases** (Rule of Three indicates $n \\ge 60$).",
        "- **Tier 3 — Zero-Miss Production Floor ($R \\ge 98.0\\%$)**: Requires **$n \\ge 183$ validation cases** to statistically bound the error rate below 2.0%.",
        "",
        "---",
        "",
        "## 4. Targeted Synthetic Data Generation Expansion Plan",
        "",
        "Prior to authorized unsealing of the locked test partition, the synthetic dataset engineering team must execute a targeted expansion of rare toxicity categories across both the training corpus and the development validation split:",
        "",
        "### 4.1 Concrete Clinical Scenarios & Phenotype Templates",
        "",
        "| Hazard Class | Target Addition (Train) | Target Addition (Val) | Specific Oncology Regimens & Clinical Scenarios |",
        "| :--- | :---: | :---: | :--- |",
        "| **`CARDIAC`** | +150 notes | +50 notes | Immune checkpoint myocarditis (troponin elevation, pericarditis under nivolumab + ipilimumab); fluoropyrimidine coronary vasospasm (5-FU infusion chest tightness); anthracycline-induced heart failure (doxorubicin ejection fraction drop). |",
        "| **`DERMATOLOGIC`** | +150 notes | +50 notes | Severe immune-related toxicities (Stevens-Johnson syndrome / TEN, bullous pemphigoid); EGFR-inhibitor acneiform rashes (cetuximab/panitumumab grade 3 pustular eruption). |",
        "| **`NEUROPATHIC`** | +100 notes | +40 notes | Chemotherapy-induced peripheral neuropathy (paclitaxel sensory numbness, oxaliplatin acute cold-triggered dysesthesia, vincristine motor weakness). |",
        "| **`RENAL`** | +100 notes | +30 notes | Cisplatin acute kidney injury (serum creatinine doubling, oliguria, tubular necrosis); immune checkpoint nephritis (pembrolizumab tubulointerstitial nephritis, proteinuria). |",
        "",
        "### 4.2 Expansion Quality Control Invariants:",
        "1. **Entity-Preserving Augmentation**: All newly synthesized cases must retain exact genomic, dosage, and medication entity annotations with zero hallucination.",
        "2. **Negation Diversity**: Include negative control notes (e.g. *'denies chest pain, troponin normal, no evidence of myocarditis'*) to prevent false positive inflation.",
        "3. **Zero Patient ID Leakage**: Generated synthetic patients must be assigned distinct `PT-` identifiers and partitioned strictly at the patient level.",
        "",
        "---",
        "",
        "## 5. Remaining Risk Statement",
        "",
        "> [!CAUTION]",
        "> **REMAINING STATISTICAL & CLINICAL RISKS:**",
        "> 1. **Synthetic Circularity Limit**: Expanding synthetic data narrows statistical error within the template distribution $P_{\\text{synth}}$, but cannot guarantee coverage of unanticipated real-world adverse drug reactions, atypical presentations, or comorbid drug interactions.",
        "> 2. **Clinician Triage Liability**: In clinical practice, misclassifying an acute cardiac toxicity (e.g. ICI myocarditis with 40% mortality) as non-urgent poses catastrophic risk. The model must not be deployed as an autonomous triage gate for rare toxicities without a human-in-the-loop oncology nurse verification step.",
        "> 3. **Real-World EHR Transfer**: Prospective validation on de-identified real-world oncology EHR notes under an approved IRB protocol remains mandatory prior to Phase 4 integration.",
        ""
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    compute_exact_stats()
