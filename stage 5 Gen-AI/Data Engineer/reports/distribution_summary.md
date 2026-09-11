# Stage 5 Reference Distribution Summary Report

**Execution Timestamp**: 2026-09-11T16:42:59.435689+00:00  
**Compiled Artifact**: `C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/data/processed/reference_distributions.parquet`  
**Distribution Version**: v1.0  

---

## 1. Genomic Mutation Frequencies (Top Driver Alterations)

| Rank | Mutation Gene | Observed Count | Empirical Frequency |
| :---: | :--- | :---: | :---: |
| 1 | **NONE/UNKNOWN** | 5,237 | 0.5982 |
| 2 | **EGFR** | 2,295 | 0.2622 |
| 3 | **KRAS** | 2,163 | 0.2471 |
| 4 | **WILD-TYPE** | 1,823 | 0.2082 |
| 5 | **TP53** | 1,635 | 0.1868 |
| 6 | **ALK** | 1,285 | 0.1468 |
| 7 | **BRAF** | 1,153 | 0.1317 |
| 8 | **ROS1** | 975 | 0.1114 |

---

## 2. Mutation Co-Occurrence & Dual Alterations

Total Mined Co-Occurrence Pairs: **36**

| Mutation A | Mutation B | Joint Count | Joint Frequency | Rarity Category |
| :--- | :--- | :---: | :---: | :---: |
| EGFR | NONE/UNKNOWN | 889 | 0.101554 | `common` |
| KRAS | NONE/UNKNOWN | 836 | 0.095499 | `uncommon` |
| NONE/UNKNOWN | TP53 | 578 | 0.066027 | `uncommon` |
| ALK | NONE/UNKNOWN | 438 | 0.050034 | `uncommon` |
| NONE/UNKNOWN | WILD-TYPE | 396 | 0.045236 | `uncommon` |
| BRAF | NONE/UNKNOWN | 377 | 0.043066 | `uncommon` |
| EGFR | WILD-TYPE | 326 | 0.037240 | `uncommon` |
| KRAS | WILD-TYPE | 284 | 0.032442 | `uncommon` |
| NONE/UNKNOWN | ROS1 | 274 | 0.031300 | `uncommon` |
| MET | NONE/UNKNOWN | 271 | 0.030957 | `uncommon` |

---

## 3. Biomarker Summary Distributions

| Biomarker | Unit | Mean ± Std | Median (P50) | Range [Min – Max] | P05 – P95 Range |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `gene_expression_score` | score | 49.7302 ± 14.6999 | 49.81 | [0.0 – 103.03] | [25.166 – 73.8935] |
| `hemoglobin` | g/dL | 12.5064 ± 1.9591 | 12.5 | [5.2 – 18.0] | [9.3 – 15.8] |
| `white_blood_cell_count` | 10^3/uL | 7.4929 ± 3.3864 | 7.43 | [0.5 – 80.0] | [2.5565 – 12.38] |
| `platelet_count` | 10^3/uL | 231.5342 ± 86.9312 | 231.0 | [10.0 – 588.0] | [85.0 – 377.0] |
| `creatinine_level` | mg/dL | 1.0329 ± 0.7255 | 1.01 | [0.2 – 15.0] | [0.36 – 1.66] |
| `liver_function_marker` | U/L | 24.9835 ± 23.797 | 17.3 | [5.0 – 245.7] | [5.0 – 72.1] |
| `systolic_bp` | mmHg | 127.7135 ± 17.8133 | 128.0 | [80.0 – 196.0] | [98.0 – 156.0] |
| `diastolic_bp` | mmHg | 79.5577 ± 11.9925 | 80.0 | [40.0 – 124.0] | [60.0 – 99.0] |
| `heart_rate` | bpm | 79.737 ± 14.2297 | 79.0 | [40.0 – 220.0] | [57.0 – 103.0] |
| `oxygen_saturation` | % | 95.891 ± 2.6641 | 96.0 | [83.2 – 100.0] | [91.3 – 100.0] |

---

## 4. Treatment & Dosage Ranges

| Treatment Regimen | Modality | Frequency | Observed Range | Median Dose |
| :--- | :--- | :---: | :---: | :---: |
| **ALECTINIB** | Systemic | 0.0022 | [10.0 – 257.5 mg] | 157.4 mg |
| **Alectinib** | Systemic | 0.0418 | [10.0 – 5000.0 mg] | 150.3 mg |
| **Atezolizumab** | Systemic | 0.0579 | [10.0 – 344.5 mg] | 150.3 mg |
| **CARBOPLATIN** | Systemic | 0.0031 | [32.7 – 272.3 mg] | 150.3 mg |
| **CARBOPLATIN+PEMBROLIZUMAB** | Systemic | 0.0019 | [40.5 – 5000.0 mg] | 124.7 mg |
| **CISPLATIN+PEMETREXED** | Systemic | 0.0032 | [98.8 – 314.8 mg] | 181.5 mg |
| **CRIZOTINIB** | Systemic | 0.0021 | [45.4 – 245.7 mg] | 127.55 mg |
| **Carboplatin** | Systemic | 0.0569 | [10.0 – 342.6 mg] | 150.3 mg |
