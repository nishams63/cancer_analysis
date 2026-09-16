from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from pathlib import Path

OUT = Path('output/pdf/Oncology_Project_Viva_Preparation.pdf')
OUT.parent.mkdir(parents=True, exist_ok=True)

navy = colors.HexColor('#12263A')
blue = colors.HexColor('#1769AA')
cyan = colors.HexColor('#EAF5FF')
light = colors.HexColor('#F5F8FB')
gold = colors.HexColor('#FFF4D6')
green = colors.HexColor('#EAF8F1')
muted = colors.HexColor('#526575')

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Title2', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=25, leading=31, textColor=navy, alignment=TA_CENTER, spaceAfter=8))
styles.add(ParagraphStyle(name='SubTitle', parent=styles['Normal'], fontSize=11.5, leading=16, textColor=muted, alignment=TA_CENTER, spaceAfter=18))
styles.add(ParagraphStyle(name='H1x', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=18, leading=23, textColor=navy, spaceBefore=7, spaceAfter=10, keepWithNext=True))
styles.add(ParagraphStyle(name='H2x', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=13.2, leading=17, textColor=blue, spaceBefore=9, spaceAfter=5, keepWithNext=True))
styles.add(ParagraphStyle(name='H3x', parent=styles['Heading3'], fontName='Helvetica-Bold', fontSize=10.8, leading=14, textColor=navy, spaceBefore=7, spaceAfter=3, keepWithNext=True))
styles.add(ParagraphStyle(name='Bodyx', parent=styles['BodyText'], fontSize=9.4, leading=13.3, textColor=colors.HexColor('#243746'), spaceAfter=5))
styles.add(ParagraphStyle(name='Smallx', parent=styles['BodyText'], fontSize=8.2, leading=11, textColor=muted, spaceAfter=3))
styles.add(ParagraphStyle(name='Q', parent=styles['BodyText'], fontName='Helvetica-Bold', fontSize=9.5, leading=13, textColor=navy, spaceBefore=5, spaceAfter=2, keepWithNext=True))
styles.add(ParagraphStyle(name='A', parent=styles['BodyText'], fontSize=9.2, leading=13, leftIndent=9, textColor=colors.HexColor('#243746'), spaceAfter=5))
styles.add(ParagraphStyle(name='Callout', parent=styles['BodyText'], fontSize=9.5, leading=13.5, textColor=navy, backColor=cyan, borderColor=blue, borderWidth=.6, borderPadding=8, spaceBefore=6, spaceAfter=8))

def P(text, style='Bodyx'):
    return Paragraph(text, styles[style])

def q(question, answer):
    return [P('Q: ' + question, 'Q'), P('A: ' + answer, 'A')]

def table(headers, rows, widths=None, small=False):
    data = [[P(str(h), 'Smallx') for h in headers]]
    for row in rows:
        data.append([P(str(c), 'Smallx' if small else 'Bodyx') for c in row])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), navy), ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BACKGROUND', (0,1), (-1,-1), colors.white), ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light]),
        ('GRID', (0,0), (-1,-1), .35, colors.HexColor('#C9D5DF')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'), ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6), ('TOPPADDING', (0,0), (-1,-1), 5), ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    return t

def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#D6E0E8'))
    canvas.line(16*mm, 14*mm, 194*mm, 14*mm)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(muted)
    canvas.drawString(16*mm, 9*mm, 'Oncology Treatment Optimization - Viva Preparation')
    canvas.drawRightString(194*mm, 9*mm, f'Page {doc.page}')
    canvas.restoreState()

story=[]
story += [Spacer(1, 18*mm), P('ONCOLOGY TREATMENT OPTIMIZATION', 'Title2'), P('Technical Interview and Viva Preparation', 'SubTitle'),
          P('<b>How to use this PDF:</b> The panel pattern is usually "what, why, compare, failure, prevention." For every answer, state the goal, explain the choice, compare one alternative, then describe validation.', 'Callout'),
          P('<b>Project summary answer:</b> We built a patient-level oncology decision-support research prototype. Clinical records and longitudinal biomarkers are prepared without patient leakage, pathology is modeled as a bag of patient tiles, NLP extracts structured context, and a local API returns typed decision-support fields with warnings and human-review boundaries.', 'Bodyx'),
          P('<b>Scientific honesty:</b> The connected Stage 4 local service is a rule-based integration baseline. The lower dashboard treatment cards are illustrative reference panels. Do not describe them as a trained, clinically validated SLM or an automatic prescription system.', 'Callout'), PageBreak()]

story += [P('THE UNIVERSAL INTERVIEW ANSWER PATTERN', 'H1x'),
          P('When the interviewer asks "Why did you use X?", answer in this order:', 'Bodyx'),
          table(['Step','What to say','Example'], [
              ('1. Goal','Name the prediction or engineering problem.','We need to identify progression cases with few missed positives.'),
              ('2. Data reality','Mention size, imbalance, modality, missingness or timing.','Patients have repeated visits and irregular biomarker measurements.'),
              ('3. Choice','Explain why X fits the constraint.','Recall is prioritized because missed progression needs review.'),
              ('4. Alternative','Acknowledge Y and its strength.','Accuracy is simple, but can hide poor minority-class recall.'),
              ('5. Evidence','Say how you would prove the choice.','Compare on patient-disjoint validation and report recall, precision, F1 and PR-AUC.'),
          ], [25*mm, 66*mm, 87*mm], small=True),
          P('Never say "X is best" without a condition. Say "X was the better choice for this constraint, provided validation confirms it."', 'Callout'),
          P('HIGH-FREQUENCY CORE QUESTIONS', 'H1x')]
story += q('Why is recall more important than accuracy in this project?', 'Accuracy treats every error equally. In progression screening, a false negative means a progressing patient may be missed, while a false positive normally causes an additional clinician review. Therefore we prioritize sensitivity/recall, but we do not ignore precision, F1, PR-AUC, calibration and workload. A model that predicts every patient positive has high recall but poor precision, so the threshold must be chosen on validation data and the tradeoff reported.')
story += q('What is the difference between raw data, cleaned data and feature data?', 'Raw data is the source record as collected: notes, dates, lab values, image metadata and missing codes. Cleaned data has corrected types, normalized units and timestamps, resolved duplicates, documented missingness and removed invalid records while preserving provenance. Feature data is the numeric or encoded representation supplied to a model, such as scaled biomarkers, one-hot treatment indicators, days from baseline, delta days, missingness masks or CNN embeddings. Cleaning improves reliability; feature engineering makes information learnable. We never fit transformations using test data.')
story += q('What do you do if the model fails?', 'First classify the failure: data/label issue, leakage, preprocessing mismatch, software/API error, distribution shift, overfitting, underfitting or an unsafe output. Reproduce it with the saved seed and split, inspect examples and confusion matrix, compare with the baseline, check train/validation curves, then make one documented change. Re-run validation and robustness checks. If the model still fails, report the failure and use the safer baseline or human review path; never edit the metric.')
story += q('How do you prevent overfitting and underfitting?', 'For overfitting, use patient-level splits, training-only preprocessing, augmentation, weight decay, dropout where appropriate, early stopping, simpler heads and independent seeds. For underfitting, check that features and labels are correct, increase capacity or training time carefully, reduce excessive regularization and improve representation. Training and validation curves distinguish the two: a widening train-validation gap suggests overfitting; both curves remaining poor suggests underfitting.')
story += q('How do you choose X model over Y model?', 'Start from data type and constraints, not popularity. Compare both under the same split, preprocessing, budget, stopping rule and metrics. Choose the model with the best validation utility, stability, calibration and latency. The test set is used once after selection. If Y wins, report Y.')

story += [PageBreak(), P('STAGE 1: MACHINE LEARNING', 'H1x'), P('Data Engineering workflow', 'H2x'),
          P('Define the target and prediction time; identify the patient as the split unit; collect records; maintain a data dictionary; normalize units and dates; resolve duplicates; split patients; fit preprocessing on training only; validate leakage; version the manifest.', 'Bodyx'),
          table(['Decision','Why','Alternative'], [
              ('Patient split','Repeated visits are correlated; prevents identity leakage.','Random row split is optimistic.'),
              ('Median + mask','Robust to skew and preserves "not measured" information.','Mean is outlier-sensitive; KNN is slower and distance-dependent; dropping rows wastes cases.'),
              ('Training-only scaler','Prevents validation/test distribution information entering training.','Fit-on-all-data is leakage.'),
              ('Day 0-90 input','Matches the forecasting question.','Full-history input leaks future observations.'),
          ], [39*mm, 76*mm, 63*mm], small=True),
          P('EDA workflow', 'H2x'), P('Run schema and quality checks, univariate distributions/missingness, bivariate feature-target analysis, multivariate correlations/interactions and subgroup checks, then temporal interval/history analysis and split-distribution audit. Univariate describes one variable; bivariate finds relationships; multivariate reveals redundancy and interactions.', 'Bodyx'),
          P('Feature engineering', 'H2x'),
          table(['Choice','Why used','Alternative'], [
              ('One-hot encoding','Nominal categories have no numeric order.','Label encoding falsely implies order.'),
              ('StandardScaler','Good for linear, distance and gradient-based models.','MinMax bounds values but is outlier-sensitive; RobustScaler handles heavy tails.'),
              ('Log transform','Reduces positive biomarker skew.','Clipping can hide clinically meaningful extremes.'),
              ('Missingness masks','Missing tests can be informative.','Imputation alone hides observation status.'),
          ], [35*mm, 78*mm, 65*mm], small=True),
          P('Model selection and evaluation', 'H2x'),
          table(['Model','Strength','Weakness / when not to use'], [
              ('Logistic Regression','Fast, interpretable baseline, useful probabilities.','Linear boundary; misses complex interactions.'),
              ('Random Forest','Robust nonlinear tabular benchmark, little scaling.','Large and less smooth; probabilities may need calibration.'),
              ('XGBoost','Strong structured-data performance and interactions.','More tuning; can overfit small cohorts.'),
              ('SVM','Good margins in high-dimensional small data.','Needs scaling; kernel cost grows with sample size.'),
          ], [36*mm, 68*mm, 74*mm], small=True),
          P('Classification metrics: accuracy for balanced costs, precision for false-alarm cost, recall for missed-positive cost, F1 for the precision-recall balance, ROC-AUC for ranking and PR-AUC for rare positives. Regression: MAE is easy to interpret, RMSE penalizes large errors and R2 describes explained variance but can be negative.', 'Bodyx')]
for item in [
    ('Why not drop missing rows?', 'Missingness is common and informative. Dropping rows reduces power and can bias the cohort. We use imputation plus a mask and validate the choice.'),
    ('Why not use KNN imputation?', 'KNN can preserve local structure, but it is slower, sensitive to scale and unstable with sparse mixed data. It is a candidate to test, not an automatic default.'),
    ('Why keep logistic regression?', 'It is a transparent sanity check. A complex model must beat a simple baseline under the same split to justify its cost.'),
    ('What is leakage through feature engineering?', 'Computing a scaler, imputation value, selected feature set or target-derived statistic using validation/test rows leaks information. Fit these objects on training only.'),
]: story += q(*item)

story += [PageBreak(), P('STAGE 2: DEEP LEARNING', 'H1x'), P('Why DL and architecture workflow', 'H2x'), P('Traditional ML is appropriate for compact tables. DL is useful for high-dimensional pathology pixels and ordered longitudinal patterns because it learns representations. The project preserves ResNet-18 and BiLSTM baselines while evaluating ResNet-50, Attention-MIL and a time-aware Transformer.', 'Bodyx'),
          table(['Data','Model choice','Why over alternatives'], [
              ('Pathology tiles','CNN; ResNet-18 baseline, ResNet-50 upgrade','CNN captures local morphology; ViT is a benchmark but generally needs more data/compute.'),
              ('12 tiles per patient','Gated Attention-MIL','Mean is stable but equal-weight; max is noise-sensitive; attention ranks influential tiles.'),
              ('Irregular visits','Time-aware Transformer','LSTM is a strong baseline; Transformer can connect distant visits when data supports it.'),
          ], [35*mm, 47*mm, 96*mm], small=True),
          P('Training workflow', 'H2x'), P('Set seed and patient split; initialize pretrained weights; train the head with the backbone frozen initially; optionally fine-tune upper layers using a smaller learning rate; use AdamW, weight decay and cosine scheduling; monitor validation; early-stop and save the best validation checkpoint; evaluate once on test.', 'Bodyx'),
          table(['Choice','Why','Alternative'], [
              ('AdamW','Fast adaptive convergence and decoupled weight decay.','SGD may generalize strongly but needs more tuning; RMSProp can help noisy recurrent gradients.'),
              ('BCE / focal loss','BCE for binary classification; focal emphasizes hard/rare examples.','MSE is not suited to a binary target.'),
              ('Huber regression loss','Less sensitive to ctDNA outliers than MSE.','MSE penalizes large misses more strongly.'),
              ('Early stopping','Stops when validation generalization degrades.','Last-epoch checkpoint may be overfit.'),
          ], [35*mm, 76*mm, 67*mm], small=True),
          P('Regularization', 'H2x'), P('Dropout removes co-adaptation, batch normalization stabilizes CNN activations, weight decay limits large weights, plausible augmentation tests invariance, and early stopping controls memorization. Excessive dropout or augmentation can cause underfitting.', 'Bodyx'),
          P('Evaluation Engineering', 'H2x'), P('Evaluate at patient level, not tile level. Report confusion matrix, Accuracy/Precision/Recall/F1/ROC-AUC/PR-AUC, ctDNA MAE/RMSE/R2, parameter count, training/inference time and mean +/- standard deviation across independent seeds. Compare with classical baselines and ablate attention, time features and task heads.', 'Bodyx')]
for item in [
    ('Why ResNet-50 over ResNet-18?', 'ResNet-50 has greater capacity and stronger pretrained representation for complex morphology, but costs more. We choose it only if patient-level validation supports the cost; otherwise the baseline remains the honest choice.'),
    ('Why not freeze a random backbone?', 'A frozen random backbone produces fixed noise features. The head cannot learn useful visual filters. Pretrained weights provide a meaningful starting representation.'),
    ('Why patient-level MIL instead of tile classification?', 'The label belongs to the patient, not necessarily every tile. MIL learns from a bag and attention makes tile influence inspectable.'),
    ('How do you identify overfitting?', 'A widening train-validation gap, unstable seeds and degraded perturbed performance are warning signs. I check curves, confusion matrix and baseline comparison before changing the model.'),
]: story += q(*item)

story += [PageBreak(), P('STAGE 3: NLP', 'H1x'), P('Text Data Engineering', 'H2x'), P('Preserve the original note and provenance; normalize encoding and whitespace; conservatively remove direct identifiers; preserve negation, dosage, units, temporal expressions and sections; tokenize with the model tokenizer; split by patient; validate empty notes and truncation.', 'Bodyx'),
          table(['Choice','Why','Alternative'], [
              ('Regex cleaning','Transparent for known IDs, whitespace and patterns.','Brittle; library tokenizers are stronger for boundaries but still need QA.'),
              ('Lemmatization','Dictionary forms preserve meaning better than suffix chopping.','Stemming is faster but can create nonwords and merge medical terms.'),
              ('Minimal normalization','Protects "no progression", dose and units.','Aggressive cleaning can reverse clinical meaning.'),
          ], [38*mm, 75*mm, 65*mm], small=True),
          P('Text representation', 'H2x'), table(['Representation','Strength','Limitation / use'], [
              ('Bag of Words','Simple counts and interpretable baseline.','Ignores order; fast sanity model.'),
              ('TF-IDF','Highlights discriminative terms.','Sparse and order-insensitive; strong small-data baseline.'),
              ('Word2Vec / GloVe','Dense semantic vectors.','Static and domain/polysemy limitations.'),
              ('Contextual embeddings','Meaning changes with context; handles phrases and negation better.','More compute, domain shift and truncation.'),
          ], [40*mm, 67*mm, 71*mm], small=True),
          P('Architecture and integration', 'H2x'), P('Classical TF-IDF plus logistic regression/SVM is the interpretable baseline. CNN text models capture local phrases, LSTM models order, and Transformers capture contextual relationships. The integration contract carries patient_id, document_id, note, preprocessing/model versions and timestamp. The typed response carries risk, finding, action, routing, safety flag, warnings and latency; the client validates it before state updates.', 'Bodyx'),
          P('Evaluation', 'H2x'), P('Use accuracy/precision/recall/F1/PR-AUC for classification, exact match or entity F1 for extraction, BLEU for translation, ROUGE for summarization overlap, perplexity for next-token likelihood, and human review for factuality, usefulness and safety. BLEU and ROUGE alone do not prove clinical correctness.', 'Bodyx')]
for item in [
    ('Why preserve negation?', '"No evidence of progression" has the opposite meaning of "progression." Cleaning must retain negation and test negated examples.'),
    ('Why not use BLEU alone?', 'BLEU measures surface overlap, not factuality, entity preservation, contradiction or safety. Structured checks and human review are required.'),
    ('How does NLP connect to the API?', 'The frontend sends a validated note contract. The service returns a typed schema. Errors, timeouts, malformed JSON and wrong-patient responses are visible and do not update state.'),
]: story += q(*item)

story += [PageBreak(), P('STAGE 4: SLM (SMALL LANGUAGE MODELS)', 'H1x'), P('Why an SLM?', 'H2x'), P('An SLM is attractive for bounded extraction, classification, summarization or routing when local data governance, predictable latency and low resource cost matter more than broad world knowledge. A full LLM may be stronger zero-shot, but can be expensive, network-dependent and harder to govern.', 'Bodyx'),
          table(['Concern','SLM','Full LLM'], [
              ('Latency/cost','Lower and more predictable.','Higher or API-variable.'),
              ('Hardware','Quantization can fit local GPU/CPU.','Often needs large GPU or hosted API.'),
              ('Coverage','Narrower; needs task data.','Broader prior knowledge.'),
              ('Governance','Easier to keep notes local.','External retention and access questions.'),
          ], [38*mm, 70*mm, 70*mm], small=True),
          P('Workflow and fine-tuning choices', 'H2x'), P('Define a narrow schema; choose a model that fits memory and context; establish a prompt-only baseline; compare full fine-tuning, LoRA and QLoRA; use patient-disjoint data, early stopping and versioned checkpoints; constrain/validate output; calibrate thresholds on validation; evaluate once on test.', 'Bodyx'),
          table(['Method','Strength','Tradeoff'], [
              ('Prompt engineering','No training cost; fast baseline.','Less consistent on domain-specific formats.'),
              ('Full fine-tuning','Maximum adaptation capacity.','Highest memory, compute and storage cost.'),
              ('LoRA','Small trainable adapters; easy to version.','May have capacity limits.'),
              ('QLoRA','Lower memory through quantized base plus adapters.','Quantization can trade quality for deployability.'),
          ], [38*mm, 72*mm, 68*mm], small=True),
          P('Data and evaluation', 'H2x'), P('Curate non-duplicated, non-contradictory, de-identified examples with valid schema, preserved negation, source and patient split. Evaluate schema-valid rate, entity F1, classification F1/PR-AUC, factuality, unsupported-claim rate, latency, peak memory, failure rate and blinded human review. Never simulate independent seeds with dropout on one checkpoint.', 'Bodyx'),
          P('Integration', 'H2x'), P('A production SLM endpoint should validate input, preprocess, run the model, validate structured output, attach version and latency, apply safety gates and return an explicit error on timeout or load failure. On a 4 GB-class GPU, quantization, small batches, gradient accumulation and truncation are resource controls, not evidence of clinical validity.', 'Bodyx')]
for item in [
    ('What does the current Stage 4 service actually do?', 'It is a deterministic rule-based decision-support baseline behind FastAPI. It validates the integration contract and safety-state behavior. It should not be described as a trained or clinically validated SLM.'),
    ('Why LoRA/QLoRA instead of full fine-tuning?', 'Adapters reduce trainable parameters, memory and checkpoint size. QLoRA reduces memory further. Full fine-tuning remains a comparison when data and hardware justify it.'),
    ('How do you prevent hallucination?', 'Use a narrow schema, evidence-only prompting or retrieval, constrained decoding, entity/contradiction checks, warnings, thresholds and human review. Fluency is not correctness.'),
]: story += q(*item)

story += [PageBreak(), P('FINAL SECTION: CROSS-STAGE COMPARISON', 'H1x'),
          table(['Area','ML','DL','NLP','SLM'], [
              ('Input','Structured tables','Images/sequences','Text','Text instructions/structured output'),
              ('Strength','Efficient/interpretable','Learns complex representations','Language/context','Local bounded language capability'),
              ('Main risk','Underfit interactions','Overfit correlated samples','Negation/domain shift','Hallucination/truncation'),
              ('Project role','Tabular baselines/outcomes','ResNet and temporal modeling','Clinical context extraction','Local integration baseline; future trained model'),
          ], [27*mm, 39*mm, 39*mm, 39*mm, 36*mm], small=True),
          P('Anticipated cross-stage "why not X?" answers', 'H2x')]
for item in [
    ('Why not random row splitting?', 'Visits, tiles and notes from one patient are correlated. Random rows let the model recognize patients and inflate metrics.'),
    ('Why not XGBoost for pathology?', 'XGBoost needs engineered image features; a CNN learns morphology directly. XGBoost remains a strong tabular comparator.'),
    ('Why not ViT instead of ResNet-50?', 'ViT models global relationships but usually benefits from more data and compute. ResNet-50 has a strong pretrained visual bias and is the fair primary upgrade.'),
    ('Why not mean pooling instead of attention?', 'Mean is stable and simple, but equal weighting can dilute a decisive tile. Attention is useful if validation shows a gain and its weights are inspected.'),
    ('Why not BiLSTM instead of Transformer?', 'BiLSTM is a strong small-data baseline. A time-aware Transformer is justified only by held-out validation, stability and resource evidence.'),
    ('Why not calibrate on test?', 'That tunes to the answer key. Fit temperature or thresholds on validation and report test calibration once.'),
    ('Why not deploy directly to clinicians?', 'The prototype needs independent external validation, prospective studies, security, monitoring, regulatory review and clinician governance. Its outputs support review; they do not prescribe.'),
]: story += q(*item)

story += [P('ROLE-WISE RAPID PREPARATION', 'H1x'),
          table(['Role','Be ready to explain'], [
              ('Data Engineering','Data dictionary, units, missingness, patient split, cutoff, lineage and reproducibility.'),
              ('EDA Engineering','Univariate, bivariate, multivariate, imbalance, outlier, temporal and subgroup findings.'),
              ('ML/DL/NLP/SLM Engineering','Input/output, baseline, loss, optimizer, regularization, model alternative and rejection criteria.'),
              ('Evaluation Engineering','Validation versus test, leakage, confusion matrix, multi-seed stability, calibration, OOD and robustness.'),
              ('Integration Engineering','UI -> validation -> API -> timeout/error -> schema check -> state update -> audit/save/export.'),
          ], [42*mm, 138*mm], small=True),
          P('Final 30-second answer', 'H2x'), P('"We built a reproducible, patient-level oncology decision-support prototype. We controlled leakage, compared simple and learned baselines, used CNN/MIL for pathology and a time-aware sequence model for longitudinal biomarkers, and connected clinical text through a typed API. We measured errors and stability rather than changing metrics. The current local Stage 4 service is rule-based, reference panels are illustrative, and clinical use would require independent validation and governance."', 'Callout')]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=16*mm, leftMargin=16*mm, topMargin=15*mm, bottomMargin=20*mm, title='Oncology Project Viva Preparation')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT.resolve())
