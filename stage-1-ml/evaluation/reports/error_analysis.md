# Detailed Error & Confidence Analysis -- Candidate V4

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Role**: Evaluation Engineer  
**Dataset**: Locked Test Set (1,750 encounters across 1,200 unique patients)  

---

## 1. Classification Error Transition Matrix

| Error Transition | Error Count | % of All Errors | Clinical Risk Interpretation |
|:---|:---:|:---:|:---|
| **Moderate -> Low** | 183 | 24.7% | Under-triage of moderate symptoms |
| **Low -> Moderate** | 209 | 28.2% | Minor false alarm (unnecessary precaution) |
| **High -> Moderate** | 74 | 10.0% | Partial detection; patient flagged for moderate monitoring |
| **Moderate -> High** | 143 | 19.3% | Over-triage; patient receives extra safety monitoring |
| **High -> Low** | 50 | 6.7% | **Critical miss**: severe toxicity patient misclassified as safe |
| **Low -> High** | 82 | 11.1% | Extreme false alarm; low risk patient flagged as critical |

---

## 2. Deep Dive: Critical High-Risk Misclassifications (High -> Low)

There were **50 critical misses** where actual High-risk encounters were predicted as Low-risk (15.0% of all High-risk encounters).

### Characteristics of Critical Miss Encounters:
- **Mean Model Confidence on Critical Misses**: 0.5097
- **Mean Prob(Low)**: 0.5097
- **Mean Prob(High)**: 0.1532
- In most critical miss cases, the model assigned substantial probability to Moderate risk, but Low slightly edged it out due to mild baseline organ impairment values.

---

## 3. Prediction Confidence Analysis

- **Mean Confidence on Correct Predictions**: **0.5553**
- **Mean Confidence on Erroneous Predictions**: **0.4958**
- **Confidence Separation**: The model is on average **0.0595** more confident when it is correct, indicating that low-confidence predictions can serve as a natural clinical uncertainty filter.

---

## 4. Class Imbalance Impact

The test set reflects real-world clinical imbalance:
- **Low**: 942 (53.8%)
- **Moderate**: 474 (27.1%)
- **High**: 334 (19.1%)

Moderate risk remains the interstitial "transition" class. When patients display mixed clinical markers (e.g. normal liver function but elevated creatinine and moderate comorbidity), the decision boundary between Moderate and Low/High is sensitive to small physiological fluctuations.
