# Empirical Stage Failure Analysis Report

## Cross-Stage Empirical Vulnerabilities

| Failure ID | Stage | Category | Error Rate | Severity | Description |
|:---|:---|:---|:---:|:---:|:---|
| `FP-001` | Stage 1 (ML) | High-Risk False Negatives | 0.4192 | **CRITICAL** | 140 high-risk toxicity cases misclassified as Moderate (77) or Low (63) due to subtle vital sign overlap. |
| `FP-002` | Stage 1 (ML) | Moderate Risk Boundary Confusion | 0.6962 | **HIGH** | Moderate risk F1 is only 0.3251; 41.8% misclassified as Low risk due to lack of distinct non-linear boundaries. |
| `FP-003` | Stage 3 (NLP) | Negation Scope Leakage | 0.0840 | **HIGH** | Sentences with contrasting clauses leak negation into affirmed symptoms ('Denies chest pain, however rapid pulse noted'). |
| `FP-004` | Stage 3 (NLP) | Minority Toxicity Over-Guessing | 0.9360 | **MEDIUM** | Aggressive inverse-class weights cause 93.6% false positive rate on DERMATOLOGIC and 54.2% error on RENAL hazards. |
| `FP-005` | Stage 4 (SLM) | Dropped Secondary Driver Alterations | 0.4800 | **HIGH** | When multiple driver mutations co-occur (e.g. EGFR + MET), SLM generation tends to omit the secondary bypass alteration. |
| `FP-006` | Stage 4 (SLM) | Out-of-Distribution Degradation | 0.0478 | **MEDIUM** | In-distribution validation ceiling (F1=1.0) drops to 0.9522 on OOD-Real progress notes with non-standard syntax. |

## Stage-Specific Vulnerability Breakdown

### Stage 1 (Tabular ML - XGBoost/CatBoost)
- **High-Risk False Negatives**: 140 patients with severe clinical toxicity misclassified as Low or Moderate risk (Recall = 58.08%).
- **Moderate Risk Confusion**: F1 score collapses to 0.3251, with 41.8% misclassified as Low.

### Stage 3 (Clinical NLP - DeBERTa)
- **Negation Leaks**: In contrastive sentences ('Denies X, however Y is present'), attention mechanism incorrectly negates Y.
- **Inverse Class Weight Over-Triggering**: Inverse frequency loss causes 93.6% false alarms in Dermatologic hazards and 54.2% in Renal hazards.

### Stage 4 (Clinical SLM - Qwen2.5-1.5B LoRA)
- **Dropped Secondary Driver Alterations**: In co-occurring genomic profiles (e.g. EGFR + MET), generation omits the bypass alteration.
- **Out-of-Distribution Degradation**: Note performance drops on messy, authentic clinic notes with heavy shorthand and dictation errors.
