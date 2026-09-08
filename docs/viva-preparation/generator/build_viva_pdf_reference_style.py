from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle

OUT = Path(r"C:\Users\nisham\Desktop\ONCOLOGY TREATMENT\output\pdf\data_science_viva_project_format.pdf")
OUT.parent.mkdir(parents=True, exist_ok=True)
BLUE = colors.HexColor('#1F2E85'); RED = colors.HexColor('#E3332B'); DARK = colors.HexColor('#34464A'); PALE = colors.HexColor('#E8EFF0')
ss = getSampleStyleSheet()
ss.add(ParagraphStyle(name='Cover', fontName='Helvetica-Bold', fontSize=18, leading=23, textColor=BLUE, alignment=TA_CENTER))
ss.add(ParagraphStyle(name='Sub', fontName='Helvetica', fontSize=9, leading=12, textColor=DARK, alignment=TA_CENTER))
ss.add(ParagraphStyle(name='Part', fontName='Helvetica-Bold', fontSize=14, leading=18, textColor=colors.white, spaceBefore=0, spaceAfter=8))
ss.add(ParagraphStyle(name='Section', fontName='Helvetica-Bold', fontSize=11, leading=14, textColor=colors.HexColor('#1765B1'), spaceBefore=4, spaceAfter=5))
ss.add(ParagraphStyle(name='Q', fontName='Helvetica-Bold', fontSize=7.3, leading=9.3, textColor=RED, spaceBefore=2.5, spaceAfter=1))
ss.add(ParagraphStyle(name='A', fontName='Helvetica', fontSize=7.1, leading=9.1, textColor=colors.black, leftIndent=4, spaceAfter=2.5))
ss.add(ParagraphStyle(name='Note', fontName='Helvetica', fontSize=7.2, leading=9.2, textColor=DARK, spaceAfter=3))

def p(s, style='A'): return Paragraph(s, ss[style])
def qa(q,a): return [p('Q: '+q,'Q'), p('A: '+a,'A')]
def bar(title, color=BLUE):
    t=Table([[p(title,'Part')]], colWidths=[16.6*cm]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),color),('LEFTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)])); return t
def role(title):
    t=Table([[p(title,'Section')]], colWidths=[16.6*cm]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),DARK),('LEFTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2)])); return t
def footer(canvas, doc):
    canvas.saveState(); canvas.setStrokeColor(colors.HexColor('#9FA9AD')); canvas.line(1.6*cm,1.25*cm,19.4*cm,1.25*cm)
    canvas.setFont('Helvetica',5.5); canvas.setFillColor(colors.HexColor('#75848A')); canvas.drawString(1.6*cm,.88*cm,'Data Science Viva Preparation Guide | Personalized Precision Medicine for Oncology Treatment Optimization'); canvas.drawRightString(19.4*cm,.88*cm,f'Page {doc.page}'); canvas.restoreState()

story=[]
story += [Spacer(1,3.6*cm), p('Data Science Viva Preparation Guide','Cover'), p('Project: Personalized Precision Medicine for Oncology Treatment Optimization','Sub'), Spacer(1,.5*cm)]
story += [Table([['']],colWidths=[10.5*cm],rowHeights=[.03*cm],style=[('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#A5BAC0'))]), Spacer(1,.6*cm), p('Covers only the project implementation: Machine Learning - Deep Learning - Data Engineer - EDA Engineer - ML Engineer - Evaluation Engineer - Integration Engineer.','Sub')]
story += [Spacer(1,2.2*cm), p('Prepared from the implemented project reports and code. All Stage 2 results are synthetic-benchmark results; they are not clinical evidence.','Sub'), PageBreak()]

story += [bar('PART 1: MACHINE LEARNING (ML) - PROJECT IMPLEMENTATION'), p('1. Stage 1: Toxicity Risk Prediction','Section')]
for q,a in [
('What was the ML problem?','A three-class supervised classification problem: predict Low, Moderate, or High oncology treatment toxicity risk.'),
('What data did the model use?','The cleaned master dataset contains 8,754 encounters from 6,000 unique patients. The raw file had 8,901 rows and 36 columns.'),
('What was the target?','The target was toxicity_risk. treatment_response was excluded from ML features because it could leak outcome information.'),
('What cleaning did you implement?','I standardized missing tokens, cleaned unit-suffixed numeric values, parsed dates, standardized categories, removed 147 duplicates, applied documented imputation, and checked plausible ranges.'),
('How did you handle missing values?','Numeric features used median imputation. Categorical values used Unknown; genomic fields used None or Unknown; prior adverse event used false.'),
('What features were used?','Thirty raw predictor fields plus 11 engineered features, including cumulative treatment load, organ impairment, vital instability, genomic instability, biomarker severity, and a comorbidity-age feature.'),
('Why was class balance important?','The labels were Low 53.4%, Moderate 27.6%, and High 19.0%. Accuracy alone would hide poor High-risk detection, so macro F1 and High-risk recall were central.'),
('How was leakage prevented?','Model selection used 5-fold StratifiedGroupKFold grouped by patient_id. Preprocessing and resampling were performed inside each fold, with zero patient overlap.'),
('Which models were compared?','The project compared LightGBM and XGBoost candidates, along with different weighting and resampling approaches.'),
('Which ML model was selected?','V4 regularized LightGBM: depth 3, 80 estimators, learning rate 0.10, min child samples 30, L1 0.5, L2 1.0, balanced weights, and conservative decision multipliers.'),
]: story += qa(q,a)
story += [PageBreak()]

story += [bar('PART 1: MACHINE LEARNING (ML) - PROJECT IMPLEMENTATION'), p('2. Training, Evaluation and Result','Section')]
for q,a in [
('Why was V4 selected over XGBoost?','The XGBoost candidate had stronger CV values but a severe generalization gap: training macro F1 0.7763 versus CV macro F1 0.5427. V4 had a smaller, more acceptable gap.'),
('Why did you not choose SMOTE?','The project found SMOTE distorted the 113-dimensional mixed one-hot and continuous feature space. It did not provide the preferred High-risk recall and stability.'),
('What validation result did V4 achieve?','Five-fold patient-grouped CV gave macro F1 0.5427 +/- 0.0041, High-risk recall 0.6444 +/- 0.0115, and accuracy 0.5889 +/- 0.0092.'),
('What was the locked-test result?','On 1,750 encounters from 1,200 held-out patients: accuracy 0.5766, macro precision 0.5209, macro recall 0.5440, macro F1 0.5288, and weighted F1 0.5766.'),
('What was the High-risk result?','High-risk recall was 0.6287 with bootstrap CI [0.5759, 0.6783]. High-risk F1 was 0.5462.'),
('Which class was hardest?','Moderate risk was hardest: its locked-test F1 was 0.3271. The error analysis showed frequent Moderate-to-Low and Moderate-to-High confusion.'),
('What uncertainty-related metrics were reported?','The locked test reported log loss 0.9129 and mean Brier score 0.1817. Correct predictions had mean confidence 0.5553 versus 0.4958 for errors.'),
('How was the final model integrated?','Frozen V4 artifacts are model.joblib, preprocessor.joblib and target mapping JSON. FastAPI exposes /health, /predict and /predict/batch.'),
('What does the API return?','Validated input is feature-engineered and preprocessed, then the API returns Low, Moderate, or High risk with class probabilities.'),
('What is the honest limitation?','This is a research prototype, not a clinical decision system. It requires external validation, clinical governance, and real-world monitoring before use.'),
]: story += qa(q,a)
story += [PageBreak()]

story += [bar('PART 1: MACHINE LEARNING (ML) - PROJECT IMPLEMENTATION'), p('3. ML Viva Answers','Section')]
for q,a in [
('Explain precision and recall in this project.','Precision asks whether predicted risk labels are correct. Recall asks how many actual cases are detected. High-risk recall matters because missing severe toxicity is clinically concerning.'),
('Why macro F1?','Macro F1 averages class-level F1 equally, so the majority Low-risk class cannot dominate the score.'),
('What is a confusion matrix?','It counts actual-versus-predicted labels. In the final test, 50 actual High-risk cases were predicted Low; this was treated as a critical miss in error analysis.'),
('What is overfitting here?','A model can look excellent on training data but fail on new patients. The rejected XGBoost candidate showed this through its large train-to-CV gap.'),
('What are feature engineering examples?','Cumulative treatment load, organ impairment, vital instability, genomic instability, biomarker severity, and a comorbidity-age feature were engineered from raw clinical variables.'),
('What is cross-validation?','The project used five patient-grouped folds to compare candidates while keeping a patient entirely in one fold.'),
('How did you make inference robust?','Pydantic validates API input. The OneHotEncoder uses handle_unknown=ignore so unseen categorical values do not crash preprocessing.'),
('What would you improve next?','Externally validate on real, diverse patient cohorts; reassess calibration and fairness; and monitor drift after any real deployment.'),
]: story += qa(q,a)
story += [PageBreak()]

story += [bar('PART 2: DEEP LEARNING (DL) - PROJECT IMPLEMENTATION'), p('1. Stage 2: Pathology and Temporal Baselines','Section')]
for q,a in [
('What is the Stage 2 task?','A synthetic multimodal NSCLC research benchmark: pathology tile classification plus longitudinal ctDNA forecasting and progression classification.'),
('What is the pathology dataset?','1,000 synthetic patients, 12,000 pathology tiles, and 12 tiles per patient. Tiles are 224 x 224 RGB images with 4,000 benign, 4,000 malignant and 4,000 inflammation tiles.'),
('What pathology model was actually trained?','The preserved baseline is a pretrained ResNet-18 transfer model with frozen backbone and a custom 512-to-128-to-3 classification head.'),
('What image preprocessing was used?','Resize to 224 x 224 and ImageNet normalization. Training used horizontal/vertical flips, 90-degree rotation, and light color jitter; validation/test used no augmentation.'),
('What temporal data was used?','16,012 biomarker observations, with 14-18 visits per patient. Features include ctDNA VAF, CEA, CA125, LDH, CRP, temporal indices, delta days, ctDNA velocity, and five missingness masks.'),
('What temporal model was actually trained?','The baseline is a 2-layer bidirectional LSTM with 13 input features, hidden size 64 in each direction, and separate ctDNA regression and progression-classification heads.'),
('Why include masks and delta days?','Masks preserve information about missing biomarkers. days_from_baseline and delta_days let the model represent irregular intervals rather than assuming equally spaced visits.'),
('How was temporal leakage prevented?','Only Day 0-90 observations were inputs. Day 91+ observations were forecasting targets. Future trajectory type and targets were prohibited from inputs.'),
('How were patients split?','700 train, 150 validation, and 150 test patients, with zero patient overlap. Every tile and temporal observation inherits its patient split.'),
]: story += qa(q,a)
story += [PageBreak()]

story += [bar('PART 2: DEEP LEARNING (DL) - PROJECT IMPLEMENTATION'), p('2. Training, Evaluation and Integration','Section')]
for q,a in [
('What were the pathology training settings?','AdamW, learning rate 0.001, weight decay 0.0001, batch size 64, cross-entropy loss, cosine scheduler, and 3 epochs.'),
('What were the temporal training settings?','AdamW, learning rate 0.002, weight decay 0.001, batch size 32, 25 epochs, Smooth L1 plus BCEWithLogits loss, and ReduceLROnPlateau.'),
('What did the held-out pathology test report?','For 1,800 tiles, accuracy, balanced accuracy, macro precision, macro recall, macro F1, and ROC-AUC were all 1.0.'),
('What did the held-out temporal test report?','For 150 patients, ctDNA MAE was 0.3698, RMSE 0.5060 and R2 0.8236. Progression accuracy, precision, recall, F1, ROC-AUC and PR-AUC were all 1.0.'),
('Can these DL scores be called clinically excellent?','No. The reports explain that the deterministic synthetic generator made the classes and trajectories unusually separable. These results validate the benchmark implementation, not clinical utility.'),
('How does multimodal fusion work?','The prototype fixed fusion is 0.35 x pathology malignant probability + 0.40 x progression probability + 0.25 x normalized ctDNA forecast.'),
('What happens with a missing modality?','The API reports FULL_MULTIMODAL, PATHOLOGY_ONLY, TEMPORAL_ONLY, or INSUFFICIENT_DATA. Temporal-only fusion uses 0.6 progression plus 0.4 ctDNA.'),
('What explainability was implemented?','Grad-CAM supports pathology-region inspection; temporal permutation importance and feature ablation support sequence diagnostics. They do not establish biology.'),
('What robustness checks exist?','The Stage 2 evaluation includes image blur and stain shifts, temporal missingness checks, confusion matrices and diagnostic error inspection.'),
]: story += qa(q,a)
story += [Table([[p('Note: The requested ResNet-50, Attention-MIL, Transformer, learned fusion, calibration, OOD and multi-seed upgrade experiments were not presented here as completed results. This guide reports only validated baseline implementation evidence.','Note')]],colWidths=[16.6*cm],style=[('BACKGROUND',(0,0),(-1,-1),PALE),('BOX',(0,0),(-1,-1),.25,colors.HexColor('#B7CBCE')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]), PageBreak()]

story += [bar('PART 3: ROLE-SPECIFIC PROJECT IMPLEMENTATION'), role('Data Engineer')]
for q,a in [
('What did you build as Data Engineer?','A reproducible CSV pipeline: raw oncology data to master_patient_dataset.csv, plus Stage 2 synthetic tile/temporal datasets and split manifests.'),
('How did you ensure data quality?','I standardized missing codes, categories, units and dates; removed duplicates; checked value bounds; documented imputation; and verified no unhandled missing values in Stage 1.'),
('What were the Stage 1 quality results?','Raw 8,901 rows became 8,754 cleaned rows after 147 duplicate removals. Around 4,200 missing-token/null instances were handled.'),
('Did you use a database?','No. The implemented project uses CSV datasets, joblib artifacts, PyTorch checkpoints, reports and API payloads.'),
('How did Stage 2 keep splits safe?','Patient split manifests were generated first, then every image tile and biomarker visit inherited the patient partition.'),
]: story += qa(q,a)
story += [role('EDA Engineer')]
for q,a in [
('What did EDA check in Stage 1?','Dataset shape/types, missingness, duplicates, class distribution, distributions, correlations and feature relationships.'),
('What did EDA find in Stage 2 images?','All 12,000 tiles were readable 224 x 224 RGB; no blanks or corrupt files. Color, brightness, contrast and stain variation were designed independently of class.'),
('What did EDA find in Stage 2 sequences?','There were no temporal inversions, controlled missingness of about 5.31%-5.60%, and no Day 90 horizon violations.'),
]: story += qa(q,a)
story += [PageBreak()]

story += [bar('PART 3: ROLE-SPECIFIC PROJECT IMPLEMENTATION'), role('ML Engineer')]
for q,a in [
('What was your ML engineering responsibility?','Build leakage-safe features, train and compare candidates, select the stable V4 model with validation evidence, freeze artifacts, and implement prediction inference.'),
('What made selection reliable?','Patient-grouped cross-validation, fold-local preprocessing/resampling, a locked test set, generalization-gap review, and bootstrap confidence intervals.'),
('Why not select only on accuracy?','Class imbalance requires macro F1 and High-risk recall. The project explicitly monitored minority-class behavior.'),
]: story += qa(q,a)
story += [role('Evaluation Engineer')]
for q,a in [
('What was your evaluation responsibility?','Evaluate frozen models on patient-disjoint test sets; produce classification, regression, confidence, error and subgroup evidence; and audit leakage and robustness.'),
('Train, validation and test roles?','Train fits parameters. Validation selects architecture and hyperparameters. Test is used once for final, unbiased performance evidence.'),
('What is data leakage?','It is information from outside the allowed training/input context that makes a model look better than it is. Examples avoided here: patient overlap, fold-wide preprocessing, treatment_response, and Day 91+ temporal observations.'),
('How did you analyze failure?','Stage 1 reviewed error transitions and critical High-to-Low misses. Stage 2 used confusion matrices, perturbation checks and diagnostic explanations.'),
]: story += qa(q,a)
story += [role('Integration Engineer')]
for q,a in [
('What did you integrate?','Frozen models, preprocessing, schema validation and FastAPI endpoints. Stage 1 has /health, /predict and /predict/batch; Stage 2 supports patient-level multimodal prototype inference.'),
('What are the Stage 2 outputs?','Progression probability, 30-day ctDNA forecast, fusion risk level, available-modality state, and diagnostic pathology/temporal scores.'),
('What would production require?','Authenticated deployment, monitoring, drift checks, external validation, governance, auditability and clinical review. These are not implemented clinical claims.'),
]: story += qa(q,a)
story += [PageBreak()]

story += [bar('PART 3: QUICK REVISION - PROJECT-SPECIFIC ANSWERS'), p('Final 5-Minute Revision Sheet','Section')]
for q,a in [
('Stage 1 in one sentence?','A patient-aware LightGBM system predicts three toxicity-risk classes from cleaned oncology encounters and is exposed through FastAPI.'),
('Stage 1 best evidence?','Locked-test macro F1 0.5288 and High-risk recall 0.6287, reported with bootstrap confidence intervals.'),
('Stage 2 in one sentence?','A synthetic multimodal research prototype combines a ResNet-18 pathology baseline with a multi-task BiLSTM biomarker baseline and fixed fusion.'),
('Most important leakage answer?','No patient overlap across splits; preprocessing is train-only; Stage 2 inputs are Day 0-90 and Day 91+ is target-only.'),
('Most important limitation?','Synthetic Stage 2 data can make metrics look unusually strong. Neither system is clinically validated.'),
('Best answer if asked about ResNet-50/Transformer?','They are planned upgrade targets, but I only claim what the validated baseline reports prove: ResNet-18 and BiLSTM results.'),
('Most important metrics?','Stage 1: macro F1 and High-risk recall. Stage 2 classification: F1/ROC-AUC/PR-AUC; regression: MAE/RMSE/R2.'),
('Core system flow?','Raw data -> cleaning and EDA -> patient-safe training -> validation selection -> locked test -> frozen artifacts/checkpoints -> API inference.'),
]: story += qa(q,a)
story += [Spacer(1,.35*cm), Table([[p('<b>Remember:</b> explain what you implemented, cite the actual metric, name the leakage safeguard, and state the limitation. Never claim synthetic benchmark performance is clinical performance.','Note')]],colWidths=[16.6*cm],style=[('BACKGROUND',(0,0),(-1,-1),PALE),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)])]

doc=SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=1.65*cm,rightMargin=1.65*cm,topMargin=1.35*cm,bottomMargin=1.55*cm)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)
