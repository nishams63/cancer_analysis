# Speech Recognition Clinical Error Analysis

## 1. Executive Summary

This report provides a detailed error audit of speech transcription failures across the 128 validation audio recordings, categorizing linguistic and acoustic confusion mechanisms that degrade clinical comprehension.

---

## 2. Taxonomy of Clinical Speech Recognition Errors

### Category A: Phonetic Medication Mangling
Consumer STT engines rely on general-language language models that heavily penalize low-frequency medical terms in favor of common English phrases.

| Intended Oncology Drug | Transcribed Acoustic Output | Phonetic Distance / Confusion Mechanism |
| :--- | :--- | :--- |
| **Cisplatin** (`/sɪsˈplætɪn/`) | *"this flat in"* (`/ðɪs flæt ɪn/`) | Fricative substitution (`/s/` $\rightarrow$ `/ð/`) + word boundary split |
| **Pemetrexed** (`/ˌpɛməˈtrɛksɛd/`) | *"perimeter set"* (`/pəˈrɪmɪtər sɛt/`) | Syllabic compression + vowel centralization |
| **Osimertinib** (`/ˌoʊsɪˈmɜːrtɪnɪb/`) | *"a simmer to nip"* (`/ə ˈsɪmər tu nɪp/`) | Schwa prefix insertion + multi-word segmentation |
| **Bevacizumab** (`/ˌbɛvəˈsɪzəmæb/`) | *"beverly scissor map"* (`/ˈbɛvərli ˈsɪzər mæp/`) | Lexical attraction to high-frequency proper nouns |
| **Carboplatin** (`/ˌkɑːrboʊˈplætɪn/`) | *"carbon plate in"* (`/ˈkɑːrbən pleɪt ɪn/`) | Bilabial stop alignment to common words |

### Category B: Alphanumeric Biomarker Fragmentation
Genetic variants are verbalized as sequences of letters and numbers, which recognizers consistently fail to bind into unified entity spans.

| Ground Truth Biomarker | Verbalized Speech | Transcribed STT Output | Downstream Consequence |
| :--- | :--- | :--- | :--- |
| `EGFR T790M` | *"E G F R T seven ninety M"* | *"easy of our T seven nine zero and"* | Biomarker tagger fails on lowercase/split tokens |
| `KRAS G12C` | *"K ras G twelve C"* | *"Kay rass G 12 see"* | Gene name split into homophones; `C` converted to verb `see` |
| `BRAF V600E` | *"B raf V six hundred E"* | *"be rough the 600 E"* | Acronym converted to adjective *"rough"* |
| `ALK` | *"A L K fusion"* | *"elk fusion"* | Phonetic merger with animal noun *"elk"* |

### Category C: Unit & Numeric Conversions
Numbers in medical dosages exhibit high sensitivity to punctuation and compounding:
- Spoken: *"seventy-five milligrams per meter squared"*
- STT Output: *"75 milligrams per meter squared"* (Text form) or *"75 mg / m 2"* (Fragmented form)
- Downstream Consequence: Regex expecting `mg/m2` failed on expanded word formats until transcript normalization was applied.

---

## 3. Recommended Remediation Architecture

To enable reliable speech-driven clinical workflows, future research must incorporate:
1. **Domain-Specific Language Model (LM) Rescoring**: Injecting an oncological n-gram or masked language model (e.g., PubMedBERT) to rescore STT lattice outputs.
2. **Clinical Lexicon Biasing**: Constraining decoder beam search with hospital formulary drug lists and biomarker dictionaries.
3. **Phonetic Entity Linking**: Post-processing STT outputs using phonetic algorithms (Double Metaphone, Soundex, or Editex) to map *"this flat in"* back to *"cisplatin"*.
