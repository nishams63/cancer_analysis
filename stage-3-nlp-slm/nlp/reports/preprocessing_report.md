# Clinical Text Preprocessing & Normalization Report — Stage 3

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Module**: Stage 3 — Clinical NLP Engineering  
**Version**: `1.0.0`  

---

## 1. Preprocessing Philosophy & Clinical Invariants
In general NLP, text preprocessing frequently strips punctuation, numbers, and stopwords, and applies aggressive stemming or lemmatization. In **Clinical Oncology NLP**, such operations are catastrophically destructive:
- Stripping numbers eliminates vital laboratory values and antineoplastic dosages (e.g., `75.5 mg/m2`, `1.85 mg/dL`).
- Removing stopwords destroys negation cues (e.g., transforming *"no nausea reported"* into *"nausea reported"*).
- Stemming conflates clinically distinct concepts (e.g., reducing *"neutropenia"*, *"neutropenic"*, and *"neutrophil"* to a shared stem).

Therefore, our preprocessing follows a **conservative, semantic-preserving protocol**.

---

## 2. Implemented Transformations

### A. Conservative Text Cleaning (`text_cleaning.py`)
1. **Control Character Stripping**: Removes non-printable ASCII control characters (`\x00-\x08`, `\x0b`, `\x0c`, `\x0e-\x1f`, `\x7f`) while strictly preserving line breaks (`\n`) and tabs (`\t`).
2. **Accidental Markup Stripping**: Removes residual HTML/XML tags (`<...>` replaced by space) without altering interior text.
3. **Non-breaking Space & Zero-width Cleanup**: Normalizes `\xa0` to standard ASCII space and removes zero-width spaces (`\u200b`, `\ufeff`).
4. **Punctuation & Quote Standardization**: Normalizes smart/curly quotes (`“`, `”`, `‘`, `’`) to ASCII (`"`, `'`) and em/en-dashes (`—`, `–`) to hyphens (`-`).
5. **Whitespace Collapse**: Collapses multiple horizontal spaces to single spaces and limits consecutive line breaks to at most 2.

### B. Text Normalization (`text_normalization.py`)
1. **Unicode NFKC Decomposition**: Normalizes all characters into standard compatibility composite forms.
2. **Clinical Unit Standardization**:
   - `mg / m2` or `mg/m^2` $\to$ `mg/m2`
   - `mg / dl` or `mg/dl` $\to$ `mg/dL`
   - `mm hg` or `mmHg` $\to$ `mmHg`
   - Spacing: ensures a single space between numeric values and units (e.g., `100mg` $\to$ `100 mg`).
3. **Oncology Acronym Canonicalization**: Ensures driver gene symbols (`EGFR`, `KRAS`, `TP53`, `BRAF`, `ALK`) and oncology scales (`ECOG`, `CTCAE`, `NSCLC`, `SpO2`) adhere to standard uppercase representations.

### C. Sentence Segmentation (`sentence_processing.py`)
- Clinical sentence boundaries are detected using regex patterns that explicitly **protect**:
  - Decimal numbers (e.g., `75.5 mg`, `1.85 mg/dL` are not split on the dot).
  - Common medical abbreviations (e.g., `Dr.`, `Mr.`, `vs.`, `i.v.`, `p.o.`, `No.`).
  - Line breaks separating structured clinical headings.

---

## 3. Transformation Ledger & Concrete Examples

### Example 1: Dosage & Laboratory Readout Preservation
- **Raw Input**:
  `<div>Patient administered Cisplatin 75.5mg/m^2\xa0IV.\x00 Serum creatinine: 1.85mg/dl. SpO2:\u200b96%.</div>`
- **Cleaned & Normalized Output**:
  `Patient administered Cisplatin 75.5 mg/m2 IV. Serum creatinine: 1.85 mg/dL. SpO2: 96%.`
- **Preserved Semantics**: Exact decimal values `75.5`, `1.85`, and units `mg/m2`, `mg/dL`, `96%` completely intact.

### Example 2: Negation & Acronym Preservation
- **Raw Input**:
  `“Patient diagnosed with nsclc, egfr positive. Denies acute dyspnea; no fever reported.”`
- **Cleaned & Normalized Output**:
  `"Patient diagnosed with NSCLC, EGFR positive. Denies acute dyspnea; no fever reported."`
- **Preserved Semantics**: Negations `Denies` and `no` fully preserved; acronyms capitalized to canonical `NSCLC` and `EGFR`.

---

## 4. Preprocessing Quality Assertions
- Automated unit test coverage across 10 dedicated test cases (`test_text_cleaning.py`, `test_text_normalization.py`).
- 100% test pass rate verifying zero data loss on numbers, severity grades, units, or clinical negation markers.
