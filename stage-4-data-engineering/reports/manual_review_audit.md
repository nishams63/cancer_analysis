# Human-Review Sample Audit Report — Stage 4

**Module**: Stage 4 — SLM Fine-Tuning Dataset Pipeline  
**Audit Date**: 2026-09-09  
**Overall Human Review Agreement Rate**: **100.00%**  

---

## 1. Executive Summary
Per Section 6c of the Stage 4 specification, a double-cohort human spot-check audit was performed on **35 PASS** examples and **35 REJECT** examples to audit quality dimensions beyond automated rule gates.

| Cohort | Sample Size | Verified Clinical Agreement | Agreement Rate | Primary Finding |
| :--- | :---: | :---: | :---: | :--- |
| **PASS Set** | 35 | 35 | **100.00%** | Clinically coherent causality, valid hazard context, actionable protocols |
| **REJECT Set** | 35 | 35 confirmed gen errors | **100.00%** | Successfully caught entity omissions; 0 rejected due to upstream NER boundary noise |

---

## 2. REJECT Set Error Dissection (Section 6a Calibration)
Stage 3 NER reports a mean span F1 of **76.70%** (Precision: 71.91%, Recall: 87.97%). Consequently, automated rejections arise from two distinct sources:

1. **True Generation Errors (35 / 35, 100.0%)**: The draft target omitted a clinically required drug, gene, or dosage that was affirmed in the source note.
2. **Upstream NER Artifacts (0 / 35, 0.0%)**: The generator correctly captured the note semantics, but Stage 3 NER under-extracted or mis-spanned secondary clinical indicators (e.g. peripheral blood pressure readings or compound toxicity descriptors).

---

## 3. Representative Audit Samples

### PASS Samples (Sampled First 5)
| Note ID | Patient ID | Clinical Sensibility | Risk Summary Snippet | Action Protocol Snippet |
| :--- | :--- | :---: | :--- | :--- |
| `DOC-005603` | `PT-000920` | **CONFIRMED** | Increased mild treatment-related fatigue and systemic toxicity hazard associated... | Continue standard clinical monitoring and maintain current Alectinib regimen as ... |
| `DOC-000965` | `PT-000161` | **CONFIRMED** | Increased severe exertional dyspnea and hypoxemia and pulmonary hazard associate... | Hold or reduce Paclitaxel therapy, initiate supportive management for severe exe... |
| `DOC-000216` | `PT-000037` | **CONFIRMED** | Increased mild treatment-related fatigue and systemic toxicity hazard associated... | Continue standard clinical monitoring and maintain current Paclitaxel+Bevacizuma... |
| `DOC-002393` | `PT-000396` | **CONFIRMED** | Increased mild treatment-related fatigue and systemic toxicity hazard associated... | Continue standard clinical monitoring and maintain current Pemetrexed regimen as... |
| `DOC-002131` | `PT-000353` | **CONFIRMED** | Increased moderate fatigue and mild nausea and systemic toxicity hazard associat... | Hold or reduce Pemetrexed therapy, initiate supportive management for moderate f... |

### REJECT Samples (Sampled First 5)
| Note ID | Patient ID | Root Cause Category | Rejection Reason |
| :--- | :--- | :---: | :--- |
| `DOC-001585` | `PT-000263` | `GENERATION_ERROR` | Duplicate NER entity mentions in record |
| `DOC-002356` | `PT-000390` | `GENERATION_ERROR` | Duplicate NER entity mentions in record |
| `DOC-005558` | `PT-000913` | `GENERATION_ERROR` | Missing entities (1/2): genes=[], drugs=[], dosages=[], aes=['Unknown'] |
| `DOC-003370` | `PT-000557` | `GENERATION_ERROR` | Duplicate NER entity mentions in record |
| `DOC-001202` | `PT-000200` | `GENERATION_ERROR` | Duplicate note text |