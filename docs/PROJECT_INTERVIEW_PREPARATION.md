# Oncology Treatment Optimization — Technical Interview Preparation

This guide explains the project as a workflow. The project is a research prototype for oncology decision support: patient clinical text and other evidence are collected, cleaned, represented, modeled, evaluated and exposed through an integration layer. It is not an autonomous prescribing system.

Use this sentence when asked to summarize the project:

> “We built a patient-level oncology decision-support pipeline. We controlled data leakage, separated train/validation/test patients, compared classical and deep models, added NLP and a lightweight local decision-support service, and exposed the result through an API with explicit safety and human-review boundaries.”

## STAGE 1: MACHINE LEARNING

### Data Engineering workflow

The correct order is:

1. Define the prediction question and the prediction time. For example, use information available through day 90 to predict a later progression event or a 30-day biomarker value.
2. Define the unit of splitting. In oncology, the patient is the unit, even when a patient has many visits, samples or images.
3. Collect source tables: demographics, diagnoses, treatment history, laboratory biomarkers, pathology metadata and outcome labels.
4. Keep a data dictionary: field meaning, unit, allowed range, missing-value code, source and timestamp.
5. Remove duplicates and resolve conflicting records according to a documented rule, such as the most recent verified value.
6. Normalize units and timestamps. Convert dates to a common timezone and compute `days_from_baseline` and `delta_days`.
7. Split patients into train, validation and test before fitting imputers, scalers or feature selectors.
8. Fit preprocessing only on training data and apply the frozen transformation to validation and test data.
9. Check leakage: no patient appears in two splits, and no observation after the forecasting cutoff enters the input features.
10. Version the dataset, preprocessing configuration, random seed and split manifest.

| Decision | Why it is useful here | Alternative and tradeoff |
|---|---|---|
| Patient-level split | Prevents the model from memorizing patient-specific patterns across visits or tiles. | Random row split is easier but gives optimistic leakage-prone results. |
| Median imputation for skewed laboratory values | Robust to extreme biomarker values and simple to reproduce. | Mean is sensitive to outliers; KNN can preserve local structure but is slower and may leak if fitted incorrectly; dropping rows loses scarce oncology cases. |
| Missingness indicator | “Not measured” can carry clinical meaning. | Imputation alone hides the distinction between an observed normal value and an absent test. |
| Last verified value for duplicate records | Keeps a traceable clinical record. | Averaging duplicates can create a value that was never measured. |
| Training-only preprocessing | Gives an honest estimate of generalization. | Fitting on all data is convenient but uses information from validation/test distributions. |
| Explicit cutoff, day 0–90 input and day 91+ target | Matches the forecasting question. | Using the full patient history creates future-information leakage. |

### EDA workflow

Run EDA in this order:

1. **Schema and quality audit:** row count, patient count, duplicate rate, data types, units and impossible ranges.
2. **Univariate analysis:** distributions, class counts, missingness percentage, outliers and log-skew. This tells us whether scaling, robust statistics or transformation is needed.
3. **Bivariate analysis:** feature versus target, correlations, box plots by outcome and missingness versus outcome. This finds signal, confounding and possible label problems.
4. **Multivariate analysis:** correlation heatmap, pair plots for a small feature subset, PCA/UMAP for inspection and subgroup analysis. This reveals redundancy, interactions and whether classes overlap in combination even when individual features look weak.
5. **Temporal EDA:** visit intervals, history length, biomarker trajectories and cutoff compliance.
6. **Split audit:** compare train/validation/test distributions without using test labels to tune the model.

Univariate analysis answers “what does one variable look like?” Bivariate analysis answers “how does one variable relate to another?” Multivariate analysis answers “what happens when variables interact?” None proves causation.

### Feature engineering

Typical engineered fields include age bands where clinically justified, treatment-line indicators, biomarker deltas, slope over time, time since treatment, visit interval, missingness masks and one-hot encoded categorical values.

| Choice | Use when | Compared with |
|---|---|---|
| One-hot encoding | Nominal categories such as treatment type or specimen source. | Label encoding falsely implies order; target encoding can leak and needs careful cross-fitting. |
| Ordinal encoding | Categories genuinely have order, such as toxicity grade. | One-hot loses the order; label encoding is wrong for nominal values. |
| StandardScaler | Approximately centered numeric features or models using distances/gradients, such as logistic regression, SVM and neural networks. | MinMaxScaler is bounded in [0,1] but is sensitive to future outliers; RobustScaler is better when heavy outliers remain. Tree models usually do not need scaling. |
| Log transform | Positive right-skewed measurements such as some biomarker concentrations. | Clipping alone hides tail behavior; a transform should be documented and applied consistently. |
| Feature selection by validation | Remove redundant or unstable predictors while preserving generalization. | Correlation filtering is fast but misses nonlinear signal; mutual information detects nonlinear dependence but can be noisy; embedded L1 or tree importance is model-dependent. |

### Model selection

| Model | Strengths | Weaknesses | Best fit |
|---|---|---|---|
| Logistic Regression | Strong interpretable baseline, fast, calibrated after scaling/regularization. | Linear decision boundary and limited interaction modeling. | Small/medium tabular data and a transparent baseline. |
| Random Forest | Handles nonlinearities, interactions, mixed scales and noisy features with little preprocessing. | Larger, less smooth, weaker extrapolation and probability calibration may need work. | Robust tabular benchmark. |
| XGBoost/gradient boosting | Often excellent on structured data; captures nonlinear interactions efficiently. | More hyperparameters, can overfit small data and requires careful validation. | Strong tabular production candidate when interpretability is managed. |
| SVM | Effective in high-dimensional spaces and with kernels; margin-based generalization. | Scaling required, kernel cost grows with data, probabilities need calibration. | Smaller datasets with a clear margin. |

The honest selection rule is validation performance plus stability, calibration, inference cost and interpretability. The test set is used once for final reporting, never for architecture selection.

### Evaluation

For binary progression classification:

- **Accuracy** = correct predictions / all predictions. Useful only when classes and error costs are reasonably balanced.
- **Precision** = true positives / predicted positives. Important when false alarms are costly.
- **Recall/sensitivity** = true positives / actual positives. In oncology, missing a progression case can be more harmful than requesting extra review, so recall is often prioritized.
- **F1** = harmonic mean of precision and recall. Useful when both matter and the class is imbalanced.
- **ROC-AUC** measures ranking across thresholds; it can look optimistic under severe imbalance.
- **PR-AUC** focuses on positive-class retrieval and is often more informative for rare progression.

For a continuous ctDNA forecast use MAE (easy clinical units), RMSE (penalizes large misses) and R² (variance explained, but it can be negative on poor extrapolation). Report confidence intervals or mean ± standard deviation across independent seeds where feasible.

### Common interview questions and model answers

**Q: Why is patient-level splitting mandatory?**  
**A:** “Repeated visits from the same patient are correlated. A random row split can put one patient’s earlier visits in training and later visits in test, so the model appears accurate because it recognizes the patient. We split by patient first and verify all pairwise intersections are empty.”

**Q: Why not drop all rows with missing values?**  
**A:** “Missingness is common and may be clinically informative. Dropping rows reduces sample size and can bias the cohort toward patients with complete testing. We use a documented imputation rule plus missingness masks, and fit it on training data only.”

**Q: Why focus on recall instead of accuracy?**  
**A:** “The cost of a false negative—missing a patient whose disease is progressing—can be higher than the cost of a false positive that triggers clinician review. We still report precision, F1, PR-AUC and calibration because a recall-only model could over-alert.”

**Q: Why keep a simple baseline?**  
**A:** “A baseline tells us whether the complex model adds value. Without logistic regression or a simple clinical rule, an impressive deep model may only be learning an easy signal.”

**Q: What is leakage through preprocessing?**  
**A:** “If I compute a mean, scaler or selected feature set using validation or test rows, information from those distributions enters training. I avoid it by splitting first, fitting preprocessing on train only and serializing that fitted pipeline.”

## STAGE 2: DEEP LEARNING

### Why deep learning was chosen

Traditional ML is strong for compact structured tables. Deep learning becomes useful when the input has high-dimensional spatial or sequential structure: pathology tiles contain morphology, and longitudinal biomarkers contain order and irregular timing. A CNN can learn visual features instead of relying on hand-engineered texture statistics. A sequence model can learn trajectory patterns instead of only using aggregate values.

The fair answer is not “deep learning is always better.” On a small tabular cohort, gradient boosting may outperform it. We use DL where representation complexity justifies it and compare against simpler baselines.

### Architecture workflow

For pathology, the workflow is:

`12 patient tiles → pretrained CNN → tile embeddings → aggregation → patient-level prediction`.

The baseline uses ResNet-18. The primary upgrade is ResNet-50, with the baseline checkpoint preserved. ResNet-50 is deeper and usually provides a stronger visual representation, at higher memory, training and inference cost. A pretrained backbone is preferable to a randomly initialized frozen backbone: frozen random convolutional filters cannot learn useful pathology features.

For patient-level aggregation, compare mean, median, max and gated Attention-MIL. Mean is stable but treats every tile equally; max can be dominated by one noisy tile; median is robust but may discard a decisive lesion tile; Attention-MIL learns weights and makes tile importance inspectable.

For temporal data:

`visits + biomarkers + missingness masks + days_from_baseline + delta_days → temporal encoder → classification head + ctDNA regression head`.

The baseline is a two-layer bidirectional LSTM. The upgrade is a Transformer with an explicit time representation, so visits are not assumed equally spaced. LSTM is a strong small-data sequence baseline and naturally processes order. Transformers model long-range relationships and parallelize better, but require more data/regularization and can overfit.

CNN versus RNN versus Transformer:

| Input | Appropriate model | Reason |
|---|---|---|
| Pathology pixels | CNN/ViT | Local spatial patterns and morphology. |
| Short ordered biomarker history | LSTM/GRU | Efficient sequential memory and good small-data baseline. |
| Longer or irregular longitudinal history | Time-aware Transformer | Attention can connect distant visits; time features encode irregular intervals. |

### Training workflow

1. Set seed, create the patient-disjoint split and load only training augmentations.
2. Initialize pretrained weights and replace the classification head.
3. Train the head with the backbone frozen initially; optionally fine-tune upper layers with a smaller learning rate.
4. Use AdamW for a practical adaptive baseline, weight decay and a scheduler such as cosine annealing.
5. Monitor validation loss and task metric, save the best validation checkpoint and stop after patience is exhausted.
6. Evaluate the selected checkpoint once on the untouched test set.

| Optimizer | Strength | Tradeoff |
|---|---|---|
| Adam/AdamW | Fast early convergence, adaptive per-parameter rates; AdamW decouples weight decay. | Can generalize worse than tuned SGD; sensitive to learning-rate choice. |
| SGD + momentum | Often strong final generalization for vision. | Needs more tuning and warm-up; slower to get started. |
| RMSProp | Useful for noisy/nonstationary gradients and some recurrent models. | Less common default for modern vision/Transformer pipelines. |

For multi-task learning use a weighted sum such as `classification_loss + λ × regression_loss`, after checking the scales. Binary cross-entropy or focal loss applies to progression classification; MSE or Huber applies to ctDNA regression. Huber is less sensitive to outliers than MSE.

### Regularization and overfitting

- **Dropout:** randomly removes activations during training; useful in heads and sequence models, but excessive dropout can underfit.
- **Batch normalization:** stabilizes intermediate activations and can speed CNN training; it is not a substitute for validation or leakage control.
- **Weight decay:** discourages overly large weights.
- **Augmentation:** brightness, contrast, blur and noise for pathology; missing visits and biomarker noise for temporal challenge testing. Augmentations must be clinically plausible and applied only to training unless intentionally testing robustness.
- **Early stopping:** stops when validation performance stops improving. It protects small datasets from memorization.

### DL Evaluation Engineering

Report the same classification and regression metrics for every architecture, along with parameter count, training time, inference time and seed. Include:

- confusion matrix to show false-positive and false-negative behavior;
- ROC and precision-recall curves;
- patient-level, not tile-level, pathology metrics;
- baseline ML comparison;
- ablations such as no attention, no time features and single-task versus multi-task;
- mean ± standard deviation across independent training seeds.

### Common interview questions and model answers

**Q: Why ResNet-50 over ResNet-18?**  
**A:** “ResNet-18 is our preserved baseline. ResNet-50 has greater representational capacity and pretrained features that may capture more complex morphology. It costs more compute, so we select it only if patient-level validation results justify the cost.”

**Q: Why not freeze a randomly initialized backbone?**  
**A:** “A frozen random backbone is fixed noise from the classifier’s perspective. The head cannot recover useful visual features. We use pretrained weights and freeze initially, then fine-tune under validation control.”

**Q: Why patient-level Attention-MIL instead of tile classification?**  
**A:** “The clinical label belongs to the patient, not necessarily every tile. MIL learns from a bag of tiles and attention provides an inspectable ranking of influential tiles.”

**Q: What is the purpose of early stopping?**  
**A:** “Training loss can continue improving after generalization worsens. Early stopping selects the checkpoint with the best validation behavior and avoids reporting the last or most overfit epoch.”

**Q: How do you know a Transformer is better than an LSTM?**  
**A:** “I do not assume it. I train both under the same patient split, preprocessing, target definition and evaluation metrics, then compare validation performance, test performance once, stability across seeds and resource cost.”

## STAGE 3: NLP

### Text data engineering

The text workflow is:

1. Preserve the original note and source metadata for auditability.
2. De-identify or mask direct identifiers where required; do not claim de-identification merely because a model ran.
3. Normalize encoding, whitespace, Unicode and common formatting noise.
4. Preserve clinically meaningful negation, dosage, units, temporal expressions and section boundaries.
5. Tokenize using the same tokenizer expected by the model.
6. Split by patient before fitting vocabulary or training embeddings.
7. Validate minimum length, empty notes, unsupported characters and truncation.

| Choice | Advantage | Risk/alternative |
|---|---|---|
| Regex cleaning | Transparent and deterministic for IDs, whitespace and known patterns. | Brittle for language variation; can accidentally remove units or negation. |
| Library clinical tokenizer | Handles punctuation, sentence boundaries and token conventions consistently. | Adds dependency/version sensitivity; still needs clinical QA. |
| Stemming | Fast suffix chopping, useful for search baselines. | Produces nonwords and can merge clinically distinct forms. |
| Lemmatization | Produces dictionary forms and usually preserves meaning better. | Slower and dependent on POS/lexicon quality. |
| Minimal normalization | Preserves dosage, negation and medical terminology. | Less compact vocabulary than aggressive cleaning. |

For oncology notes, “no evidence of progression” must not become “progression.” Clinical NLP cleaning should be conservative.

### Text representation

| Representation | Strength | Limitation | Fit |
|---|---|---|---|
| Bag of Words | Simple, interpretable counts. | Ignores order and semantics; high-dimensional. | Fast baseline/classification. |
| TF-IDF | Downweights common words and highlights discriminative terms. | Still sparse and order-insensitive. | Strong small-data baseline and search. |
| Word2Vec | Dense local-context word vectors. | One vector per word; weak for polysemy and rare terms. | General semantic similarity. |
| GloVe | Global co-occurrence representation. | Static vectors and domain mismatch risk. | Transferable baseline when resources are limited. |
| Contextual embeddings | Meaning changes with context; better for negation and clinical phrases. | More compute, domain shift and token-length constraints. | Clinical entity/relation and decision-support text. |

The choice depends on dataset size, latency and interpretability. Start with TF-IDF/logistic regression as a sanity baseline; use contextual embeddings only when the task benefits and validation supports the cost.

### Model architecture

- Classical TF-IDF + logistic regression/SVM is fast and interpretable.
- CNN text models capture local n-gram patterns.
- LSTM/GRU models order and longer context but train sequentially.
- Transformers use attention and contextual representations, but need more compute and careful truncation.

For the prototype’s local service, the implemented Stage 4 path is a deterministic rule-based decision-support service behind FastAPI. It extracts a small set of entities and routes a response with guardrail fields. It should be described honestly as a lightweight integration baseline, not as a fine-tuned language model or a validated five-agent clinical model.

### Integration Engineering

The handoff contract should specify:

`patient_id`, `document_id`, original/normalized note, tokenizer version, truncation policy, extracted entities, model version and timestamp.

The service returns a structured response containing risk, key finding, action, routing, safety flag, warnings and latency. The client validates the response before updating state, handles non-2xx and malformed responses, times out requests and never silently applies a response for a different patient.

### Evaluation

| Metric | Use |
|---|---|
| Accuracy/precision/recall/F1 | Classification or entity-label decisions; report per class for imbalance. |
| ROC-AUC/PR-AUC | Threshold-independent ranking; PR-AUC is preferable for rare positives. |
| BLEU | Machine translation or local n-gram overlap; weak as a clinical quality measure by itself. |
| ROUGE | Summarization overlap with reference summaries; supplement with factuality checks. |
| Perplexity | Language-model next-token likelihood; useful for model comparison, not a direct clinical utility metric. |
| Exact match/token F1 | Extraction or slot-filling tasks. |
| Human review | Clinical usefulness, factuality, safety, clarity and unsupported claims. |

### Common interview questions and model answers

**Q: Why preserve negation?**  
**A:** “Negation changes the clinical meaning. Removing ‘no’ from ‘no progression’ reverses the label. We use conservative normalization and test negated examples explicitly.”

**Q: Why not use only BLEU for generated clinical text?**  
**A:** “BLEU measures surface n-gram overlap, not factual correctness or safety. We need entity preservation, contradiction checks, structured fields and human clinical review.”

**Q: How does NLP connect to the rest of the system?**  
**A:** “It consumes a versioned note contract and returns structured entities and warnings. The integration layer validates schema and patient identity before the downstream fusion or UI state is updated.”

## STAGE 4: SLM (SMALL LANGUAGE MODELS)

### Why an SLM instead of a full LLM

An SLM can be deployed locally with lower memory, lower latency, predictable cost and easier data governance. A full LLM generally has broader knowledge and stronger zero-shot behavior but costs more, may require network access and can be harder to control.

| Concern | SLM | Full LLM |
|---|---|---|
| Latency | Lower and more predictable. | Higher and variable. |
| Hardware | Fits a local GPU/CPU after quantization. | Often needs large GPU memory or an API. |
| Cost | Lower per request. | Higher infrastructure/API cost. |
| Domain breadth | Narrower; needs curated task data. | Broader prior knowledge. |
| Governance | Easier to keep data local. | External API/data-retention questions. |
| Failure mode | May miss rare language or overfit task format. | Can hallucinate confidently and be harder to constrain. |

The right answer is task-specific: use an SLM for bounded extraction, classification, summarization or routing when a larger model’s extra capability is not worth the latency and governance cost.

### SLM workflow

1. Define the narrow task and output schema before choosing a model.
2. Select a base model that fits available memory and supports the required language/context length.
3. Establish a prompt-only baseline.
4. Compare full fine-tuning, LoRA and QLoRA. Full fine-tuning updates every weight and is expensive; LoRA trains low-rank adapters with fewer trainable parameters; QLoRA quantizes the base model and trains adapters with lower memory.
5. Use patient-disjoint train/validation/test data, early stopping and checkpoint versioning.
6. Add constrained decoding or schema validation for structured clinical output.
7. Calibrate/threshold the final decision on validation data and evaluate once on test.

For this repository, do not claim that a fine-tuned SLM was trained unless the training run, checkpoint and evaluation artifacts exist. The currently connected local Stage 4 service is rule-based and is useful for integration testing. A defensible future upgrade is a quantized SLM behind the same validated API contract.

### SLM data engineering

Curate examples with source, patient split, task type, input, expected structured output and safety notes. Remove duplicates, contradictory labels, copied boilerplate and examples with unresolved identifiers. Filter for minimum quality, valid JSON/schema, preserved negation and entity consistency.

Use instruction formatting such as:

```text
System: Extract only supported clinical facts. Do not invent treatment advice.
User: <clinical note>
Assistant: {"risk": ..., "key_finding": ..., "action": ..., "warnings": [...]}
```

Keep test patients and near-duplicate notes out of training. Measure truncation and class balance. Do not train on future observations relative to the prediction cutoff.

### Evaluation Engineering

Compare the SLM with:

- a deterministic/rule baseline;
- TF-IDF + logistic regression for classification;
- a larger model or external reference only under the same held-out cases and prompt/task definition.

Use task metrics: accuracy/F1/PR-AUC for classification, exact match/entity F1 for extraction, MAE/RMSE for numeric forecasts, schema-valid rate, latency, peak memory and failure rate. For generated text add factuality, unsupported-claim rate, entity preservation and blinded human review. Report mean ± SD over independent seeds where training is stochastic. Never manufacture “multi-seed” results by toggling dropout on one checkpoint.

### Integration Engineering

The API should accept a validated request, run preprocessing, invoke the local model, validate the structured output, attach model/version/latency metadata, apply safety gates and return a typed response. Timeouts, model-load failures, malformed output and unsupported input must become explicit errors. The frontend should show loading, success and failure states and preserve the user’s local case.

On a 4 GB-class GPU, quantization, small batches, gradient accumulation, sequence truncation and cached embeddings can make experiments practical. These are resource controls, not evidence that the model is clinically accurate.

### Common interview questions and model answers

**Q: Why not use a full LLM?**  
**A:** “The task is bounded and latency/data locality matter. We would first prove that an SLM meets task metrics and safety checks; if it does not, a larger model may be justified. The choice is evidence- and constraint-driven.”

**Q: LoRA versus full fine-tuning?**  
**A:** “Full fine-tuning has maximum capacity but high memory and storage cost. LoRA updates a small adapter, is cheaper and easier to version, and is a sensible first experiment. QLoRA reduces memory further through quantization, with a possible quality tradeoff.”

**Q: How do you prevent hallucination?**  
**A:** “Use a narrow schema, retrieval or supplied evidence only, constrained decoding, entity-preservation checks, contradiction/guardrail rules, confidence thresholds and human review. A fluent sentence is not proof of correctness.”

**Q: What does the current Stage 4 service actually do?**  
**A:** “The connected local service is a rule-based decision-support baseline behind FastAPI. It is useful for testing request/response handling and safety-state behavior. We should not call it a trained or clinically validated SLM until a reproducible training and evaluation artifact exists.”

## FINAL SECTION: CROSS-STAGE COMPARISON

| Area | Machine Learning | Deep Learning | NLP | SLM |
|---|---|---|---|---|
| Input | Structured/tabular features | Images and complex sequences | Text tokens/documents | Text instructions and structured outputs |
| Typical strength | Efficient, interpretable baselines | Learns spatial/temporal representations | Models language structure and meaning | Local, bounded language capability |
| Data need | Low to medium | Medium to high, transfer learning helps | Depends on representation/task | Curated task examples; adapters reduce cost |
| Compute | Low | High | Low to high | Tunable; quantization enables local deployment |
| Main risk | Underfitting nonlinear interactions | Overfitting and leakage through correlated samples | Negation/domain shift | Hallucination, truncation and narrow coverage |
| Project role | Tabular baseline and outcome modeling | ResNet pathology and temporal modeling | Clinical entity/context extraction | Lightweight local decision-support integration baseline; future SLM target |

### Anticipated “why not X instead of Y?” questions

**Why not random row splitting?** Patient records are correlated; it inflates performance.

**Why not accuracy alone?** It can hide missed progression cases under imbalance; use recall, precision, F1, PR-AUC and calibration.

**Why not KNN imputation everywhere?** It is slower, distance-sensitive and can be unstable with mixed scales/missingness; use it only if validation proves benefit over a robust simpler rule.

**Why not XGBoost instead of deep learning for pathology?** XGBoost needs engineered image features. A CNN learns morphology directly; XGBoost remains a valuable tabular baseline.

**Why not ViT instead of ResNet-50?** ViT can model global relationships but usually benefits from more data and compute. ResNet-50 has strong pretrained inductive bias and is a fair primary upgrade; ViT is a benchmark.

**Why not mean pooling instead of Attention-MIL?** Mean is stable and interpretable as an average, but it gives every tile equal weight. Attention-MIL can identify decisive tiles; it must earn that complexity on patient-level validation.

**Why not BiLSTM instead of a Transformer?** BiLSTM is an important small-data baseline. The Transformer is justified only if time-aware validation shows better performance or useful long-range modeling at acceptable cost.

**Why not equal-spacing visits?** Irregular intervals change the meaning of a biomarker change. Include `days_from_baseline`, `delta_days` and missingness rather than pretending visits are equally spaced.

**Why not clean text aggressively?** Clinical negation, doses and units are signal. Conservative normalization protects meaning.

**Why not BLEU for clinical generation?** Overlap does not guarantee factuality, entity correctness or safety.

**Why not prompt engineering alone for an SLM?** Prompting is a cheap baseline and may be enough for a narrow task, but adapters can improve consistency when held-out task data supports it. We compare them experimentally.

**Why not calibrate on test data?** That tunes to the answer key. Fit temperature or thresholds on validation and report test calibration once.

**Why not claim the model is clinically ready?** The project is a research prototype using limited/synthetic or illustrative evidence, local rule-based decision support and human-review boundaries. Clinical deployment requires external validation, prospective studies, regulatory review, security, monitoring and clinician governance.

## Role-wise preparation

### Data Engineering

Be ready to explain the data dictionary, patient split manifest, units, missingness, cutoff, reproducibility seed and why preprocessing is fitted on train only.

### EDA Engineering

Be ready to show one univariate, one bivariate and one multivariate finding, plus how you checked class imbalance, outliers, temporal irregularity and subgroup differences.

### ML/DL/NLP/SLM Engineering

Be ready to justify each model against a baseline, explain the loss and optimizer, identify the main overfitting risk, describe the input/output tensor or schema, and state what evidence would make you reject your preferred model.

### Evaluation Engineering

Be ready to explain why the validation set selects models, the test set reports final performance, patient-level splits prevent leakage, multi-seed runs measure stability, and robustness/calibration/OOD tests reveal failure modes hidden by F1.

### Integration Engineering

Be ready to trace one request from UI to API and back: validation → loading state → request timeout/error handling → response schema validation → state update → audit/save/export. Mention that unavailable integrations are surfaced explicitly rather than silently returning fake success.

## A strong final answer to “what did you contribute?”

> “My role was to make the oncology pipeline reproducible and reviewable. I separated patient-level data splits from preprocessing, compared baselines with learned models, preserved the ResNet-18 and BiLSTM baselines while evaluating the ResNet-50 and time-aware Transformer upgrades, treated pathology as a patient-level MIL problem, and connected the NLP/decision-support layer through a typed API. I reported limitations honestly: the local Stage 4 service is rule-based, lower-stage cards are illustrative, and clinical deployment would require independent validation and governance.”
