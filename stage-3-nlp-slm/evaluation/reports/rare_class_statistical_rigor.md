# Statistical Rigor in Rare Toxicity Hazards: Exact Binomial Intervals, Power Sizing, and Targeted Data Plan

**Evaluation Cohort:** Official Validation Set (`validation.parquet`, $N = 909$ clinical documents)  
**Statistical Invariant Methodologies:** Clopper-Pearson Exact Binomial & Wilson Score Inversion  
**Epistemic Scope:** **Simulation Uncertainty vs. Clinical Reality Clarification Formally Established**  

---

## 1. Executive Summary & Epistemic Principle

In clinical oncology triage, toxicity hazard classification identifies organ-specific adverse drug reactions to antineoplastic therapies. While the promoted MiniLM Hybrid model reports **100.0% validation recall** across several high-liability toxicity categories (`CARDIAC` 5/5, `DERMATOLOGIC` 4/4, `NEUROPATHIC` 12/12, and `HEMATOLOGIC` 23/23), sample sizes for these cohorts are acutely limited ($n < 30$).

> [!IMPORTANT]
> **EPISTEMIC BOUNDARY: SIMULATION UNCERTAINTY vs. CLINICAL UNCERTAINTY**
> Generating additional synthetic patient records narrows **simulation variance and estimation error within the generative model prior $P_{\text{synth}}$ only**. It **does NOT reduce real-world clinical uncertainty**, nor does it bound the domain distribution shift between template-based synthetic notes and heterogeneous hospital electronic health records $P_{\text{real}}$.
> 
> **Defensible Regulatory Stance:** While growing synthetic sample size $n$ establishes statistical robustness within the simulation benchmark, **prospective validation on institutional clinical records under an approved Institutional Review Board (IRB) protocol remains strictly mandatory** before deploying this system in clinical oncology practice.

---

## 2. Rare Hazard Class Exact Statistical Analysis ($n < 30$)

The table below contrasts empirical point estimates against **Clopper-Pearson exact 95% confidence intervals** and **Wilson score intervals** for all hazard classes on `validation.parquet` ($N=909$):

| Hazard Toxicity Class | Validation Support ($N$) | Observed TP | Empirical Recall | Clopper-Pearson 95% CI | Wilson Score 95% CI | CI Width (CP) | Statistical Confidence Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **DERMATOLOGIC** *(Rare, $n < 30$)* | 4 | 4 | 100.0% | `[0.3976, 1.0000]` | `[0.5101, 1.0000]` | 60.2% | **Extremely Low** (Width $\ge 50\%$) |
| **CARDIAC** *(Rare, $n < 30$)* | 5 | 5 | 100.0% | `[0.4782, 1.0000]` | `[0.5655, 1.0000]` | 52.2% | **Extremely Low** (Width $\ge 50\%$) |
| **NEUROPATHIC** *(Rare, $n < 30$)* | 12 | 12 | 100.0% | `[0.7354, 1.0000]` | `[0.7575, 1.0000]` | 26.5% | **Low** (Width $20-49\%$) |
| **HEMATOLOGIC** *(Rare, $n < 30$)* | 23 | 23 | 100.0% | `[0.8518, 1.0000]` | `[0.8569, 1.0000]` | 14.8% | **Defensible** (Width $< 20\%$) |
| **RENAL** *(Rare, $n < 30$)* | 24 | 22 | 91.7% | `[0.7300, 0.9897]` | `[0.7415, 0.9768]` | 26.0% | **Low** (Width $20-49\%$) |
| **PULMONARY** *(Adequate)* | 55 | 55 | 100.0% | `[0.9351, 1.0000]` | `[0.9347, 1.0000]` | 6.5% | **Defensible** (Width $< 20\%$) |
| **HEPATIC** *(Adequate)* | 76 | 62 | 81.6% | `[0.7103, 0.8955]` | `[0.7142, 0.8870]` | 18.5% | **Defensible** (Width $< 20\%$) |
| **NONE** *(Adequate)* | 710 | 698 | 98.3% | `[0.9707, 0.9912]` | `[0.9707, 0.9903]` | 2.1% | **Defensible** (Width $< 20\%$) |

### Key Statistical Findings on Rare Toxicity Hazards:
1. **Dermatologic Toxicity ($n=4$)**: Observing 4/4 correct detections yields a point estimate of 100.0%, but the exact Clopper-Pearson 95% confidence interval spans **`[0.3976, 1.0000]`** (an uncertainty span of **60.2 percentage points**). The true recall could be as low as 39.8% while still generating 4/4 detections in 5% of validation samples.
2. **Cardiac Toxicity ($n=5$)**: Point recall is 100.0%, but the exact 95% CI lower bound is **`0.4782`** (Wilson: `0.5655`). A claim of 100% recall is statistically underpowered.
3. **Neuropathic Toxicity ($n=12$)**: Point recall is 100.0%, with Clopper-Pearson 95% CI **`[0.7354, 1.0000]`**. While stronger, the lower bound still admits a 26.5% error rate.
4. **Renal Toxicity ($n=24$)**: Point recall is 91.7% (22/24), with exact 95% CI **`[0.7397, 0.9878]`**.

---

## 3. Power Analysis & Minimum Sample Size Sizing

To establish a statistically defensible claim for rare toxicity classes, we calculate the minimum sample size $n$ required under two standard binomial formulations:

### Mathematical Formulations:
1. **Rule of Three (Hanley & Lippman-Hand, 1983)**: If 0 errors are observed in $n$ cases, the 95% upper confidence bound on the true error rate is $p_{\text{err}} \le 3 / n$. Thus, to guarantee recall $\ge 1 - \epsilon$ with 95% confidence, $n \ge 3 / \epsilon$.
2. **Exact Clopper-Pearson Inversion**: For $k = n$ successes, the exact one-sided 95% lower bound satisfies $(0.05)^{1/n} = R_{\text{target}}$, yielding $n \ge \frac{\ln(0.05)}{\ln(R_{\text{target}})}$. For two-sided 95% confidence, $n \ge \frac{\ln(0.025)}{\ln(R_{\text{target}})}$.

| Target Recall Lower Bound ($R_{\text{target}}$) | Rule of Three ($n_{\min}$) | Exact Clopper-Pearson (1-sided 95%) | Exact Clopper-Pearson (2-sided 95%) | Required Validation Expansion Factor |
| :---: | :---: | :---: | :---: | :---: |
| **80%** | $n \ge 16$ | $n \ge 14$ | $n \ge 17$ | 4.2x (from $n=4$) |
| **85%** | $n \ge 20$ | $n \ge 19$ | $n \ge 23$ | 5.8x (from $n=4$) |
| **90%** | $n \ge 31$ | $n \ge 29$ | $n \ge 36$ | 9.0x (from $n=4$) |
| **95%** | $n \ge 60$ | $n \ge 59$ | $n \ge 72$ | 18.0x (from $n=4$) |
| **98%** | $n \ge 150$ | $n \ge 149$ | $n \ge 183$ | 45.8x (from $n=4$) |
| **99%** | $n \ge 300$ | $n \ge 299$ | $n \ge 368$ | 92.0x (from $n=4$) |

### Operational Recommendations for Sample Sizing:
- **Tier 1 — Minimum Defensible Benchmark ($R \ge 90.0\%$)**: Requires at least **$n = 35$ validation cases** (current Derm is $4$, Cardiac is $5$).
- **Tier 2 — Safety-Critical Benchmark ($R \ge 95.0\%$)**: Requires at least **$n = 72$ validation cases** (Rule of Three indicates $n \ge 60$).
- **Tier 3 — Zero-Miss Production Floor ($R \ge 98.0\%$)**: Requires **$n \ge 183$ validation cases** to statistically bound the error rate below 2.0%.

---

## 4. Targeted Synthetic Data Generation Expansion Plan

Prior to authorized unsealing of the locked test partition, the synthetic dataset engineering team must execute a targeted expansion of rare toxicity categories across both the training corpus and the development validation split:

### 4.1 Concrete Clinical Scenarios & Phenotype Templates

| Hazard Class | Target Addition (Train) | Target Addition (Val) | Specific Oncology Regimens & Clinical Scenarios |
| :--- | :---: | :---: | :--- |
| **`CARDIAC`** | +150 notes | +50 notes | Immune checkpoint myocarditis (troponin elevation, pericarditis under nivolumab + ipilimumab); fluoropyrimidine coronary vasospasm (5-FU infusion chest tightness); anthracycline-induced heart failure (doxorubicin ejection fraction drop). |
| **`DERMATOLOGIC`** | +150 notes | +50 notes | Severe immune-related toxicities (Stevens-Johnson syndrome / TEN, bullous pemphigoid); EGFR-inhibitor acneiform rashes (cetuximab/panitumumab grade 3 pustular eruption). |
| **`NEUROPATHIC`** | +100 notes | +40 notes | Chemotherapy-induced peripheral neuropathy (paclitaxel sensory numbness, oxaliplatin acute cold-triggered dysesthesia, vincristine motor weakness). |
| **`RENAL`** | +100 notes | +30 notes | Cisplatin acute kidney injury (serum creatinine doubling, oliguria, tubular necrosis); immune checkpoint nephritis (pembrolizumab tubulointerstitial nephritis, proteinuria). |

### 4.2 Expansion Quality Control Invariants:
1. **Entity-Preserving Augmentation**: All newly synthesized cases must retain exact genomic, dosage, and medication entity annotations with zero hallucination.
2. **Negation Diversity**: Include negative control notes (e.g. *'denies chest pain, troponin normal, no evidence of myocarditis'*) to prevent false positive inflation.
3. **Zero Patient ID Leakage**: Generated synthetic patients must be assigned distinct `PT-` identifiers and partitioned strictly at the patient level.

---

## 5. Remaining Risk Statement

> [!CAUTION]
> **REMAINING STATISTICAL & CLINICAL RISKS:**
> 1. **Synthetic Circularity Limit**: Expanding synthetic data narrows statistical error within the template distribution $P_{\text{synth}}$, but cannot guarantee coverage of unanticipated real-world adverse drug reactions, atypical presentations, or comorbid drug interactions.
> 2. **Clinician Triage Liability**: In clinical practice, misclassifying an acute cardiac toxicity (e.g. ICI myocarditis with 40% mortality) as non-urgent poses catastrophic risk. The model must not be deployed as an autonomous triage gate for rare toxicities without a human-in-the-loop oncology nurse verification step.
> 3. **Real-World EHR Transfer**: Prospective validation on de-identified real-world oncology EHR notes under an approved IRB protocol remains mandatory prior to Phase 4 integration.
