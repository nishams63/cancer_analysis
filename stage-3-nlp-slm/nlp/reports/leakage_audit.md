# Data Leakage & Split Integrity Audit Report — Stage 3 NLP

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Module**: Stage 3 — Clinical NLP Engineering  
**Version**: `1.0.0`  
**Audit Status**: ZERO LEAKAGE CERTIFIED & VERIFIED  

---

## 1. Executive Summary & Audit Matrix
To ensure that all downstream modeling is defensible and leakage-free, an exhaustive audit was performed evaluating patient overlap, encounter overlap, temporal boundaries, vocabulary fitting, and locked test isolation.

| Leakage Dimension | Verification Method | Measured Result | Audit Status |
| :--- | :--- | :---: | :---: |
| **Patient Overlap (Train $\cap$ Val)** | Set intersection of `patient_id` | **0 patients (0.0%)** | PASSED |
| **Patient Overlap (Train $\cap$ Test)**| Set intersection of `patient_id` | **0 patients (0.0%)** | PASSED |
| **Patient Overlap (Val $\cap$ Test)** | Set intersection of `patient_id` | **0 patients (0.0%)** | PASSED |
| **Encounter Overlap Across Splits** | Set intersection of `encounter_id` | **0 encounters (0.0%)**| PASSED |
| **Cross-Split Document Duplication**| Exact SHA256 text match across splits | **0 documents (0.0%)** | PASSED |
| **Preprocessing & TF-IDF Fitting** | Pipeline `.fit()` audit on TRAIN only | **TRAIN-only** | PASSED |
| **Temporal Sequence Inversions** | $\text{document\_date} > \text{index\_date}$ check | **0 records (0.0%)** | PASSED |
| **Prospective Outcome Terms** | Regex scan for 7 forbidden future phrases | **0 matches (0.0%)** | PASSED |
| **Locked Test Protection** | Code inspection of model training scripts | **100% Isolated** | PASSED |
| **Upstream Source Data Invariance** | SHA256 checksum verification | **100% Identical** | PASSED |

---

## 2. Patient & Encounter Partition Isolation
- **Train Partition**: 700 unique patients (`PT-000001` through `PT-000700`), 1,425 encounters, 4,261 documents.
- **Validation Partition**: 150 unique patients (`PT-000701` through `PT-000850`), 303 encounters, 909 documents.
- **Locked Test Partition**: 150 unique patients (`PT-000851` through `PT-001000`), 310 encounters, 928 documents.
- **Leakage Verification**:
  $$\text{Patients}(\text{Train}) \cap \text{Patients}(\text{Val}) = \emptyset$$
  $$\text{Patients}(\text{Train}) \cap \text{Patients}(\text{Test}) = \emptyset$$
  $$\text{Patients}(\text{Val}) \cap \text{Patients}(\text{Test}) = \emptyset$$
  *Automated test*: `test_split_integrity.py` passes 100%.

---

## 3. Preprocessing & Vocabulary Leakage Audit
- **Rule**: No feature transformer, vectorizer, scaler, or encoder may observe validation or test texts during fitting.
- **Implementation Verification**:
  - `TfidfVectorizer.fit()` is called exclusively on `df_train["text"]`.
  - `StandardScaler.fit()` is called exclusively on `struct_train_df`.
  - `TargetLabelManager.fit()` is called exclusively on `df_train["urgency_level"]` and `df_train["hazard_type"]`.
  - Inverse class weights $w_c$ are calculated exclusively from training label distributions.
- **Artifacts Verified**: All joblib artifacts stored in `stage-3-nlp-slm/nlp/artifacts/` reflect training-set statistics only.

---

## 4. Temporal Validity & Outcome Leakage Audit
- **Semantics**: For every clinical document, `index_date` is defined as the encounter observation date.
- **Temporal Check**:
  $$\forall d \in \text{Dataset}, \quad \text{document\_date}(d) \le \text{index\_date}(d)$$
  Across all 6,098 documents, 0 future-dated notes exist.
- **Prospective Outcome Disclosures**: An exhaustive scan for 7 forbidden terms (`retrospective survival`, `post-mortem`, `autopsy`, `progression on day 180`, `subsequent progression`, `future relapse`, `overall survival reached`) confirmed **0 occurrences**.

---

## 5. Locked Test Set Protection Certification
- **Certification**: The locked test partition (`locked_test.parquet`, 928 documents) was **NEVER** evaluated during baseline model training, hyperparameter tuning, feature thresholding, or vocabulary selection.
- **Allowed Access**: The locked test set was loaded solely by unit tests to verify schema conformance and confirm zero patient leakage against Train and Validation.
- **Integrity**: The locked test set remains an uncompromised, virgin evaluation benchmark for final project delivery.

---

## 6. Upstream Read-Only Invariance Checksums
Cryptographic SHA-256 hashes of all upstream Data Engineering processed files were verified before and after execution:

| Upstream File | Expected SHA-256 Hash | Post-NLP Measured Hash | Invariance Status |
| :--- | :--- | :--- | :---: |
| `clinical_nlp_dataset_v1.parquet` | `426ea0c51f354a5fa9e1177ec0d8fabc0ed26ed708f748ee8f8f4b0f18554230` | `426ea0c51f354a5fa9e1177ec0d8fabc0ed26ed708f748ee8f8f4b0f18554230` | UNCHANGED |
| `train.parquet` | `5862db7ec4e42fb71bf207906c9acf21f9d43280a32336ebf43dff77e6f55133` | `5862db7ec4e42fb71bf207906c9acf21f9d43280a32336ebf43dff77e6f55133` | UNCHANGED |
| `validation.parquet` | `2b6787a394d262c8f8b94b92914777f495e30a7be4eaf925ce862f1302899a4c` | `2b6787a394d262c8f8b94b92914777f495e30a7be4eaf925ce862f1302899a4c` | UNCHANGED |
| `locked_test.parquet` | `491562c892a6af2f08695eefefe782daeae12d52f1fef5a6ad11060d96af4509` | `491562c892a6af2f08695eefefe782daeae12d52f1fef5a6ad11060d96af4509` | UNCHANGED |
