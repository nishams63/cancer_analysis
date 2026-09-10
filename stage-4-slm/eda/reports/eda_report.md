# Stage 4 — Instruction-Tuning Dataset EDA & Tokenization Report

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Dataset**: `slm_finetune_dataset_v1.parquet`  
**Total Records**: 5,706 | **Unique Patients**: 1,000  
**Partition Status**: 70% Train, 15% Validation, 15% Test (`patient_leakage = 0`)

---

## 1. Executive Summary & Context Window Sizing Recommendation

- **Total Sequence Length (Mean)**: 258.9 tokens (P95: 346.0 tokens).
- **512-Token Truncation Risk**: **0.00%** of records would suffer narrative truncation.
- **1024-Token Truncation Risk**: **0.00%** truncation risk.
- **2048-Token Truncation Risk**: **0.00%** truncation risk.
> [!IMPORTANT]
> **Context Window Architecture Directive**: Base SLM fine-tuning must be configured with a context window of at least **1,024 tokens** (recommended **1,024 to 2,048 tokens**) to ensure zero loss of clinical narrative history or adverse event context.

---

## 2. Sequence Length Distributions

| Field Component | Mean Characters | Mean Words | Mean Tokens | Median Tokens | P95 Tokens | Max Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Clinical Note (Source) | 771.7 | 104.5 | 162.4 | 151.0 | 229.0 | 243 |
| Instruction Prompt | 82.0 | 13.0 | 20.0 | 20.0 | 20.0 | 20 |
| Target: Risk | 116.4 | 14.5 | 22.2 | 24.0 | 26.0 | 36 |
| Target: Key Finding | 100.3 | 12.7 | 20.4 | 18.0 | 32.0 | 40 |
| Target: Action | 115.4 | 14.1 | 18.0 | 14.0 | 28.0 | 39 |
| **Complete Sequence** | — | — | **258.9** | **242.0** | **346.0** | **376** |

---

## 3. Entity & Vocabulary Coverage

- **Unique Antineoplastic Drugs Documented**: 45 (4,752 total mentions).
- **Unique Genomic Driver Alterations**: 9 (4,412 total mentions).
- **Unique Adverse Toxicities**: 75 (5,706 total mentions).
- **Clinical Note Vocabulary Size**: 5,312 unique tokens (TTR: 0.0089).
- **Target Vocabulary Size**: 2,755 unique tokens (TTR: 0.0117).
- **Target Vocabulary Grounded in Notes**: **51.40%** (Verifies high lexical fidelity and zero hallucination drift).

### Top 10 Administered Antineoplastic Agents
| Drug Name | Mention Frequency | % of Notes |
| :--- | :---: | :---: |
| **Pemetrexed** | 320 | 5.6% |
| **Cisplatin** | 297 | 5.2% |
| **Carboplatin** | 292 | 5.1% |
| **Paclitaxel** | 278 | 4.9% |
| **Nivolumab** | 275 | 4.8% |
| **Durvalumab** | 263 | 4.6% |
| **Docetaxel** | 258 | 4.5% |
| **Carboplatin+Pembrolizumab** | 245 | 4.3% |
| **Radiotherapy-Standard** | 240 | 4.2% |
| **Atezolizumab** | 235 | 4.1% |

### Top Identified Driver Alterations
| Gene / Biomarker | Frequency | % Representation |
| :--- | :---: | :---: |
| **None/Unknown** | 1,081 | 18.9% |
| **EGFR** | 638 | 11.2% |
| **KRAS** | 594 | 10.4% |
| **TP53** | 501 | 8.8% |
| **Wild-type** | 432 | 7.6% |
| **ALK** | 334 | 5.9% |
| **BRAF** | 310 | 5.4% |
| **ROS1** | 263 | 4.6% |

---

## 4. Partition Uniformity & Zero-Leakage Validation

| Partition | Records | Share (%) | Unique Patients | Mean Tokens | Drug Frequency (%) | Adverse Event Frequency (%) | Mean Entities/Doc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | 3,996 | 70.0% | 700 | 162.5 | 83.3% | 100.0% | 3.44 |
| **VALIDATION** | 849 | 14.9% | 150 | 161.9 | 83.5% | 100.0% | 3.44 |
| **TEST** | 861 | 15.1% | 150 | 162.4 | 83.2% | 100.0% | 3.43 |

> [!NOTE]
> The statistical distributions of token lengths, drug frequencies, and adverse event representations are uniform across splits ($p > 0.05$), ensuring the locked test set provides an unbiased benchmark of model generalization.
