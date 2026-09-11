# Scenario Priority and Test Allocation Report

## Scenario Prioritization Matrix

| Scenario ID | Blind Spot | Priority Score | Tier | Recommended Cases | Target Vulnerability |
|:---|:---|:---:|:---:|:---:|:---|
| `PROMPT-R12` | `BS12` | 0.8150 | **CRITICAL** | 60 | Inverse weighting false positive on benign Grade 1 xerosis/proteinuria |
| `PROMPT-R09` | `BS09` | 0.7700 | **CRITICAL** | 55 | Sarcopenic elderly normal creatinine masking low CrCl (<30 mL/min) |
| `PROMPT-R14` | `BS14` | 0.7690 | **CRITICAL** | 50 | Stale clinic note claiming stable disease conflicting with RECIST progression |
| `PROMPT-R02` | `BS02` | 0.7490 | **CRITICAL** | 60 | Negation leakage across contrastive conjunctions (however/but) |
| `PROMPT-R10` | `BS10` | 0.7130 | **HIGH** | 35 | Adenosquamous carcinoma misclassified as conventional adenocarcinoma |
| `PROMPT-R04` | `BS04` | 0.7050 | **CRITICAL** | 50 | Decision boundary collapse in borderline moderate-risk biomarker envelope |
| `PROMPT-R15` | `BS15` | 0.6850 | **HIGH** | 40 | Subjective complaint bias ignoring silent severe hypercalcemia (14.6 mg/dL) |
| `PROMPT-R11` | `BS11` | 0.6800 | **CRITICAL** | 45 | Re-challenging checkpoint inhibitor after Grade 3 immune myocarditis |
| `PROMPT-R01` | `BS01` | 0.6550 | **CRITICAL** | 50 | Omission of MET bypass resistance in EGFR-mutant NSCLC |
| `PROMPT-R05` | `BS05` | 0.6400 | **HIGH** | 45 | Failure to catch Cisplatin contraindication during acute renal injury |
| `PROMPT-R03` | `BS03` | 0.6350 | **HIGH** | 40 | Sub-centimeter primary lesion overriding metastatic Stage IVA label |
| `PROMPT-R07` | `BS07` | 0.6150 | **HIGH** | 40 | Imputation collapse under >60% missing tabular clinical fields |
| `PROMPT-R08` | `BS08` | 0.6110 | **HIGH** | 35 | Quadruple mutation STK11/KEAP1 primary resistance to IO monotherapy |
| `PROMPT-R13` | `BS13` | 0.5950 | **MEDIUM** | 30 | OOD noisy clinic EHR note with heavy shorthand and typos |
| `PROMPT-R06` | `BS06` | 0.5890 | **HIGH** | 40 | Hyper-acute septic shock decompensation within 6 hours post-chemo |

**Total Recommended Synthetic Stress Scenarios**: 675 cases.
