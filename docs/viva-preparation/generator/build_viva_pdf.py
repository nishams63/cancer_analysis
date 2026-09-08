from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
                                KeepTogether, HRFlowable)
from reportlab.pdfbase.pdfmetrics import stringWidth

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "pdf" / "data_science_viva_project_implementation.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

NAVY = colors.HexColor("#123047")
TEAL = colors.HexColor("#087E8B")
GOLD = colors.HexColor("#E9A820")
PALE = colors.HexColor("#EDF5F6")
INK = colors.HexColor("#202B33")
MUTED = colors.HexColor("#596773")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleX", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22,
    leading=27, textColor=NAVY, alignment=TA_CENTER, spaceAfter=8))
styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=11,
    leading=15, textColor=MUTED, alignment=TA_CENTER, spaceAfter=14))
styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=16,
    leading=20, textColor=NAVY, spaceBefore=10, spaceAfter=7))
styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11.5,
    leading=14, textColor=TEAL, spaceBefore=8, spaceAfter=4))
styles.add(ParagraphStyle(name="BodyX", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.8,
    leading=12.2, textColor=INK, spaceAfter=5))
styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.6,
    leading=10, textColor=INK, spaceAfter=2))
styles.add(ParagraphStyle(name="Viva", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=8.7,
    leading=12, textColor=NAVY, backColor=PALE, borderColor=colors.HexColor("#B8D9DC"), borderWidth=.4,
    borderPadding=6, spaceBefore=3, spaceAfter=7))
styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=8.6,
    leading=11.5, textColor=colors.HexColor("#6A4700"), backColor=colors.HexColor("#FFF5D8"), borderColor=GOLD,
    borderWidth=.4, borderPadding=6, spaceBefore=5, spaceAfter=7))

def P(text, style="BodyX"):
    return Paragraph(text, styles[style])

def bullets(items):
    return KeepTogether([P("- " + x, "BodyX") for x in items])

def table(headers, rows, widths=None, small=True):
    data = [[P(h, "Small") for h in headers]] + [[P(str(x), "Small" if small else "BodyX") for x in row] for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("GRID", (0,0), (-1,-1), .25, colors.HexColor("#BFD0D7")),
        ("BACKGROUND", (0,1), (-1,-1), colors.white), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F6FAFB")]),
        ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    return t

def concept(concept, did, why, result, answer):
    return KeepTogether([
        P(concept, "H2x"),
        table(["What we did", "Why", "Result"], [[did, why, result]], [6.0*cm, 5.1*cm, 5.4*cm]),
        P("Viva answer: " + answer, "Viva")
    ])

def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CAD9DE")); canvas.line(1.6*cm, 1.25*cm, 19.4*cm, 1.25*cm)
    canvas.setFont("Helvetica", 7.5); canvas.setFillColor(MUTED)
    canvas.drawString(1.6*cm, .8*cm, "Data Science Viva - Project Implementation | Synthetic-data research prototype")
    canvas.drawRightString(19.4*cm, .8*cm, f"Page {doc.page}")
    canvas.restoreState()

story = []
story += [Spacer(1, 2.0*cm), P("DATA SCIENCE VIVA - PROJECT IMPLEMENTATION", "TitleX"),
          P("Level 1: ML -> DL -> Engineering Roles", "Subtitle"),
          HRFlowable(width="65%", thickness=2, color=TEAL, spaceAfter=14),
          P("Purpose", "H2x"),
          P("This revision guide describes only the work implemented in the <b>Personalized Precision Medicine for Oncology Treatment Optimization</b> repository. It is designed for the question: <i>What did you do in your project?</i>"),
          P("Important limitation", "Callout"),
          P("Both stages use synthetic data. Stage 1 is a toxicity-risk decision-support prototype; Stage 2 is a multimodal pathology and biomarker forecasting prototype. Results are technical benchmark results, not clinical evidence or treatment advice.", "Callout"),
          Spacer(1, .4*cm), P("One-sentence project answer", "H2x"),
          P("I built an end-to-end oncology research prototype: I cleaned and analyzed synthetic patient data, trained and independently evaluated a patient-level toxicity-risk ML model, built image and longitudinal biomarker deep-learning pipelines, and connected frozen models to validated FastAPI inference services.", "Viva"), PageBreak()]

story += [P("1. Machine Learning - Toxicity Risk Prediction", "H1x"),
          P("<b>Problem:</b> predict treatment toxicity risk as <b>Low, Moderate, or High</b> from a patient encounter. This is a <b>supervised multiclass classification</b> problem."),
          table(["Dataset", "Input", "Target", "Split"], [["8,754 cleaned synthetic encounter records from 6,000 patients", "30 raw fields, then 41 domain features / 113 transformed features", "toxicity_risk: Low / Moderate / High", "Patient-aware 80% training / 20% locked test; test = 1,750 encounters from 1,200 patients"]], [4.0*cm, 5.2*cm, 3.3*cm, 5.0*cm]),
          concept("Data preparation", "Removed 147 duplicate rows; standardized missing tokens, units, categories and dates; flagged implausible values; used median numeric imputation, explicit Unknown categories, and conservative False imputation for prior adverse events.", "Models require consistent, valid inputs and should not learn formatting errors.", "Master dataset: 8,754 rows, 35 columns, zero unhandled missing values and zero exact duplicates.", "I first converted raw oncology records into a clean, reproducible master dataset. I did not drop patient information blindly - I used a documented strategy for each type of missing value."),
          concept("Feature engineering", "Used 30 raw predictors plus 11 engineered clinical features, including cumulative treatment load, organ impairment index, vital instability score, genomic instability score, biomarker severity weight, and age-comorbidity interaction.", "Interactions summarize clinically related signals without using the target or identifiers.", "The expanded 41-feature set achieved validation macro F1 of 0.5417 before final decision-rule selection.", "Feature engineering means creating useful model inputs from available variables. I used only row-level clinical transforms and excluded patient ID, encounter ID, date, treatment response, and the target to prevent leakage."),
          concept("Models compared", "Compared tuned LightGBM V1, XGBoost V3 with thresholds, and regularized LightGBM V4. Also evaluated 12 class-balance strategies: unweighted, class weights, moderate-weight variants, asymmetric loss, random over/under-sampling, SMOTE, and out-of-fold decision adjustment.", "A comparison is more reliable than choosing one algorithm by assumption; safety required protecting high-risk recall.", "V3 overfit (train-validation gap 0.2665). No balancing candidate met all V5 promotion criteria, so V4 was retained.", "I compared models under patient-aware cross-validation. The best model was not simply the one with one high number; I balanced macro F1, high-risk recall, stability, and train-validation gap."),
          concept("Training and split", "Used 5-fold StratifiedGroupKFold on patient_id in training data. Preprocessing, class weights, and resampling were calculated inside each fold. The locked test set was untouched during training and selection.", "The same patient's encounters must not appear in both train and validation/test groups.", "Zero patient overlap was verified in all five folds and between train/test.", "I split by patient, not only by row. That protects against data leakage from repeated encounters for the same patient."),
          PageBreak()]

story += [P("Stage 1 ML - Selection and Result", "H1x"),
          table(["Candidate", "Method", "CV macro F1", "High-risk recall", "Train/val gap", "Decision"], [
              ["V1", "Tuned LightGBM, depth 6", "0.5348", "0.6009", "N/A", "Superseded"],
              ["V3", "XGBoost, depth 5 + aggressive thresholds", "0.5427", "0.6459", "0.2665", "Rejected: overfitting"],
              ["V4", "Regularized LightGBM, depth 3, 80 trees", "0.5427 +/- 0.0041", "0.6444 +/- 0.0115", "0.0950 +/- 0.0075", "Selected and frozen"],
          ], [1.4*cm, 4.5*cm, 3*cm, 3*cm, 3*cm, 3.4*cm]),
          concept("Why V4", "Used max_depth=3, n_estimators=80, learning_rate=0.10, min_child_samples=30, L1=0.5, L2=1.0, class_weight=balanced, and gentle decision multipliers [1.0, 1.05, 1.05].", "Regularization reduced the large V3 generalization gap while retaining high-risk sensitivity.", "Locked-test macro F1 0.5288, high-risk recall 0.6287, accuracy 0.5766; all with 95% bootstrap confidence intervals.", "V4 was selected because it generalized better. It was less complex than V3, had a much smaller train-validation gap, stable folds, and preserved detection of high-risk toxicity."),
          table(["Locked-test metric", "Result", "How to say it"], [
              ["Macro F1", "0.5288 [0.5035, 0.5521]", "Balances F1 across Low, Moderate and High classes, so majority Low-risk cases do not dominate."],
              ["High-risk recall", "0.6287 [0.5759, 0.6783]", "The model identified about 63% of actual high-risk encounters; this safety-oriented metric matters because misses are costly."],
              ["Accuracy", "0.5766 [0.5520, 0.6000]", "The overall proportion of correct predictions, interpreted alongside class-sensitive metrics."],
              ["Mean Brier score", "0.1817", "A probability-quality measure; lower is better."],
          ], [3.5*cm, 3.8*cm, 9.5*cm]),
          P("Error analysis", "H2x"),
          P("The evaluator examined actual errors. There were 50 critical <b>High -> Low</b> misses (15% of High-risk encounters). Mean confidence was lower on errors (0.4958) than correct predictions (0.5553), so low confidence can flag cases for extra review. Moderate risk was the hardest transition class."),
          P("Viva answer: The final Stage 1 model is frozen V4 regularized LightGBM. Its locked-test macro F1 was 0.5288 and high-risk recall was 0.6287. It is a research decision-support model, so outputs need clinician review.", "Viva"), PageBreak()]

story += [P("2. Deep Learning - Pathology and Biomarker Forecasting", "H1x"),
          P("Deep learning was used for inputs that are naturally unstructured or sequential: pathology images and longitudinal biomarker histories. The implemented frozen baseline models are described below. The repository also contains upgrade modules, but no final upgraded benchmark is presented here because it has not completed an independently verified experimental run."),
          table(["DL task", "Input", "Output", "Frozen baseline"], [
              ["Pathology image classification", "224 x 224 RGB pathology tile", "Benign / malignant / inflammation", "ResNet-18 transfer model"],
              ["Longitudinal forecasting", "Day 0-90 history; 13 biomarker/time/mask features", "30-day ctDNA forecast + progression probability", "2-layer BiLSTM with dual heads"],
          ], [4.0*cm, 4.7*cm, 4.5*cm, 3.6*cm]),
          concept("Pathology CNN", "Used torchvision ResNet-18 with transfer weights. The frozen convolutional backbone feeds a custom head: 512 -> Linear(128) -> ReLU -> Dropout(0.3) -> Linear(3). A 4-stage custom CNN is also implemented as a fallback architecture.", "Transfer learning reuses learned visual edge and texture features while the smaller head learns the three synthetic pathology classes.", "Locked test: 1,800 tiles; accuracy, macro precision, macro recall, macro F1 and macro ROC-AUC were all 1.0 on this synthetic benchmark.", "For pathology I used a ResNet-18 transfer-learning classifier. The backbone extracts image features and my head maps them to three tissue classes. The perfect score is a synthetic-data result, not clinical accuracy."),
          concept("Temporal BiLSTM", "Used a 2-layer bidirectional LSTM: 13 inputs, hidden size 64 in each direction, dropout 0.2. The representation comes from each patient's last valid historical timepoint. It has one regression head and one binary classification head.", "BiLSTM processes ordered histories; bidirectionality helps model relations across the observed history, while last-valid-step selection avoids padded tokens becoming the patient representation.", "Locked test: ctDNA MAE 0.3698, RMSE 0.5060, R2 0.8236; progression accuracy/F1/ROC-AUC/PR-AUC = 1.0 on synthetic data.", "The temporal model learns from biomarker change over time. It produces both a numerical ctDNA forecast and a progression-risk score from the same historical sequence."),
          concept("DL preprocessing", "Images: resize 224 x 224, ImageNet normalization for pretrained ResNet-18, training-only horizontal/vertical flips, 90-degree rotations and light color jitter. Temporal: training-only standardization, forward fill then zero fill at baseline, and five missingness masks.", "Normalizing stabilizes training. Training-only augmentation improves image robustness without contaminating evaluation. Masks retain information that a lab value was missing.", "Validation and test were unaugmented; training normalization parameters were reused for validation/test.", "I treated train and test differently on purpose: augmentation and fitting the scaler belong only to training, while validation and test must remain clean evaluations."), PageBreak()]

story += [P("Stage 2 DL - Training, Evaluation and Caveat", "H1x"),
          table(["Setting", "Pathology CNN", "Temporal BiLSTM"], [
              ["Optimizer", "AdamW", "AdamW"],
              ["Learning rate", "0.001", "0.002"],
              ["Weight decay", "0.0001", "0.001"],
              ["Batch size", "64", "32"],
              ["Epochs", "3", "25"],
              ["Loss", "Cross-entropy", "Smooth L1 + BCEWithLogits, lambda=1"],
              ["Scheduler", "CosineAnnealingLR", "ReduceLROnPlateau"],
          ], [4.1*cm, 6.0*cm, 6.0*cm]),
          P("Leakage controls", "H2x"),
          bullets(["1,000 patients split 700 train / 150 validation / 150 test with zero patient overlap.",
                   "Temporal inputs were limited to Day 0-90. Future Day 91+ observations were targets only.",
                   "ctDNA velocity used only the current and previous historical visit.",
                   "The normalizer was fitted on train patients only; targets were not input columns."]),
          P("How to explain the DL results", "H2x"),
          table(["Question", "Viva-ready answer"], [
              ["Why are the DL metrics so high?", "The evaluator audited them and concluded the deterministic synthetic generator makes the classes and trajectories unusually separable. I would never claim these numbers show clinical effectiveness."],
              ["How did you evaluate?", "I used a held-out, patient-disjoint test cohort: 1,800 tiles and 150 patients. For images I used accuracy, precision, recall, F1 and ROC-AUC. For ctDNA I used MAE, RMSE and R2; for progression I used accuracy, precision, recall, F1, ROC-AUC and PR-AUC."],
              ["What did robustness/explainability add?", "The evaluation module includes confusion matrices, Grad-CAM examples, temporal permutation importance, and perturbation checks such as image blur/stain shifts and temporal missingness. These are diagnostic tools, not clinical proof."],
          ], [5.2*cm, 10.9*cm]),
          P("Viva answer: Stage 2 has two separate learning paths - a CNN for pathology tiles and a multi-task BiLSTM for biomarker histories. Their outputs are fused only as prototype engineering scores, with explicit safeguards for missing modalities and future-data leakage.", "Viva"), PageBreak()]

story += [P("3. Data Engineering", "H1x"),
          P("No database was implemented. The project uses reproducible CSV-based datasets, serialized model artifacts, checkpoints, reports, and API payloads."),
          table(["What I did", "Why", "Technology used", "Viva answer"], [
              ["Stage 1: loaded raw oncology_uncleaned.csv and created master_patient_dataset.csv", "Create one validated source for EDA and ML", "Python, pandas, numpy, CSV pipeline", "I built a reproducible cleaning pipeline instead of manually editing the dataset."],
              ["Standardized missing tokens, category variants, units and dates", "Raw data had inconsistent representations", "pandas and custom validation", "I converted different formats into a consistent schema before modeling."],
              ["Removed duplicates, checked bounds and imputed documented missing values", "Prevent duplicated samples and implausible values from biasing models", "pandas/numpy validation rules", "I found 147 duplicates and removed them; the final ML dataset had no unhandled missing values."],
              ["Stage 2: generated/processed pathology tiles and longitudinal biomarker data", "Build a controlled multimodal benchmark", "Python, PIL, pandas, NumPy", "I created image and time-series pipelines with patient-level split inheritance."],
              ["Created deterministic train/validation/test manifests", "Prevent data leakage between patients", "CSV split manifests and tests", "Every tile and timepoint inherits its patient split, so no patient appears in two partitions."],
          ], [4.0*cm, 4.2*cm, 3.5*cm, 4.4*cm]),
          P("Data flow", "H2x"),
          table(["Stage", "Actual flow"], [
              ["Stage 1", "Raw CSV -> cleaning/validation -> master patient CSV -> EDA -> patient-aware ML -> frozen joblib model/preprocessor -> FastAPI response"],
              ["Stage 2", "Synthetic tiles + biomarker CSVs -> processing/split manifests -> EDA -> ResNet-18 and BiLSTM checkpoints -> aggregation/fusion -> FastAPI/dashboard/HTML report"],
          ], [3.2*cm, 12.9*cm]), PageBreak()]

story += [P("4. EDA Engineer", "H1x"),
          table(["What I checked", "Why", "What I found", "What I did next"], [
              ["Stage 1 shape and types", "Understand usable columns", "8,754 clean rows, 35 columns; 20 numerical, categorical fields, IDs and date", "Defined predictors, target and excluded identifiers."],
              ["Missing values and duplicates", "Avoid silent data-quality errors", "Raw data had about 4,200 missing-token/null instances and 147 duplicate rows; processed data had none unhandled", "Applied field-specific imputation and deduplication."],
              ["Target distribution", "See class imbalance", "Low 53.4%, Moderate 27.6%, High 19.0%", "Used macro F1, class-weight evaluation, and high-risk recall."],
              ["Distributions, correlation and feature relationships", "Find range errors, skew and potential relationships", "Created biomarker distributions, correlation heatmaps and statistical summaries", "Used these checks to guide cleaning and feature selection; not to claim causation."],
              ["Stage 2 image audit", "Verify files and shortcut risks", "12,000 readable 224x224 RGB tiles; balanced 4,000 per class; no blank/corrupt tiles", "Used standard CNN input sizing and balanced cross-entropy."],
              ["Stage 2 temporal audit", "Verify chronology, missingness and horizon separation", "16,012 observations; 0 inversions; 5.31%-5.60% controlled missingness; 0 Day 90 boundary violations", "Used masks, time features and historical-only sequences."],
          ], [3.7*cm, 3.5*cm, 4.6*cm, 4.3*cm]),
          P("Important EDA findings", "H2x"),
          bullets(["Stage 2 pathology classes were exactly balanced. Color, brightness, contrast and stain factors were audited as independent of class; this was intended to reduce shortcut learning.",
                   "Stage 2 input windows contained 7,229 historical observations (Day 0-90); 8,783 later observations were kept as future data.",
                   "The EDA reports explicitly warn that synthetic patterns can make models look stronger than real clinical data would." ]),
          P("Viva answer: EDA was not just graphs. I profiled structure, quality, missingness, duplicates, outliers, class balance, time ordering, leakage and potential visual shortcuts. The findings directly decided my preprocessing and evaluation safeguards.", "Viva"), PageBreak()]

story += [P("5. ML Engineer and 6. Evaluation Engineer", "H1x"),
          table(["Role", "My actual responsibility", "Evidence"], [
              ["ML Engineer", "Prepared leakage-safe features; trained LightGBM/XGBoost candidates; used patient-aware CV; compared regularization/class balancing/decision rules; froze V4 artifacts; implemented inference.", "V4 configuration, model.joblib, preprocessor.joblib, target mapping, feature importance, training report."],
              ["DL Engineer", "Built ResNet-18 tile classifier and BiLSTM multi-task forecaster; handled image transforms, sequence padding, missing masks, checkpoints and inference.", "best_pathology_cnn.pt, best_temporal_lstm.pt, training curves, DL tests."],
              ["Evaluation Engineer", "Evaluated frozen models on locked tests; produced metrics, bootstrap confidence intervals, confusion matrices, subgroup and error analyses; audited leakage and DL robustness.", "Stage 1 final evaluation, Stage 2 evaluation report, figures and CSV metrics."],
              ["Integration Engineer", "Connected frozen models to FastAPI schemas/endpoints and Stage 2 multimodal aggregation/fusion; handled invalid inputs and missing modalities.", "API tests, /health, /predict, /predict/batch and Stage 2 patient inference pipeline."],
          ], [3.0*cm, 7.7*cm, 5.4*cm]),
          P("Metrics actually used", "H2x"),
          table(["Metric", "What it measures", "Why this project used it"], [
              ["Accuracy", "Fraction predicted correctly", "Simple overall summary; insufficient alone for imbalanced risk classes."],
              ["Precision", "Of a predicted class, how many were correct", "Helps interpret false alarms."],
              ["Recall", "Of actual cases, how many were found", "High-risk recall was safety-critical in Stage 1."],
              ["F1 / Macro F1", "Harmonic mean of precision/recall; macro averages class scores equally", "Primary Stage 1 score because Low risk is more common."],
              ["ROC-AUC / PR-AUC", "Ranking discrimination across thresholds", "Used for binary progression and image evaluations."],
              ["MAE / RMSE / R2", "Regression error magnitude / squared-error sensitivity / explained variation", "Used for 30-day ctDNA forecasting."],
              ["Log loss / Brier", "Probability quality", "Used in Stage 1 to assess probabilistic predictions."],
              ["Bootstrap confidence intervals", "Uncertainty around a metric estimate", "Used for Stage 1 locked-test macro F1, high-risk recall and accuracy."],
          ], [3.1*cm, 5.0*cm, 8.0*cm]),
          P("Viva answer: As an evaluation engineer, I kept test data separate from model selection. I used macro F1 for balanced multiclass judgment, high-risk recall for the safety concern, and error analysis to see where the model fails.", "Viva"), PageBreak()]

story += [P("7. Integration Engineer - How the System Connects", "H1x"),
          P("Stage 1 and Stage 2 are separate project stages. Both connect their frozen artifacts to application-facing inference APIs."),
          table(["Stage 1 API", "Actual behavior"], [
              ["GET /health", "Confirms model artifacts are loaded."],
              ["POST /predict", "Validates a 30-field patient encounter, engineers/preprocesses features, runs frozen V4, returns Low/Moderate/High plus probabilities."],
              ["POST /predict/batch", "Runs the same frozen pipeline for a list of encounters."],
          ], [4.5*cm, 11.6*cm]),
          table(["Stage 2 integration", "Actual behavior"], [
              ["Pathology", "Runs tile predictions and aggregates a patient's tiles by mean, median or max."],
              ["Temporal", "Processes <=90-day biomarker history and produces progression probability plus 30-day ctDNA VAF."],
              ["Fusion", "Prototype weighted score: 0.35 x malignant probability + 0.40 x progression probability + 0.25 x normalized ctDNA risk."],
              ["Missing data", "FULL_MULTIMODAL, PATHOLOGY_ONLY, TEMPORAL_ONLY or INSUFFICIENT_DATA; no score when neither modality exists."],
              ["Application", "FastAPI endpoints, Streamlit dashboard and standalone HTML patient report are implemented."],
          ], [4.5*cm, 11.6*cm]),
          P("Simple architecture", "H2x"),
          table(["Input", "Processing", "Model", "Output"], [[
              "Raw CSV / pathology tiles / biomarker history", "Cleaning, EDA, split checks, preprocessing", "V4 LightGBM OR ResNet-18 + BiLSTM", "Risk class/probabilities OR pathology class + ctDNA/progression + prototype alert"
          ]], [3.8*cm, 4.3*cm, 4.2*cm, 4.0*cm]),
          P("Viva answer: The API does not retrain the model. It validates input, applies the saved feature/preprocessing steps, calls a frozen model, and returns a structured response. In Stage 2 it can combine pathology and temporal outputs, while clearly marking partial data.", "Viva"), PageBreak()]

story += [P("8. Complete Project Flow", "H1x"),
          table(["Stage", "What we actually did"], [
              ["1. Data", "Used synthetic oncology encounter data for Stage 1, and synthetic pathology tiles plus biomarker trajectories for Stage 2."],
              ["2. Data Engineering", "Cleaned/validated CSV data; standardized values; created processed datasets and deterministic patient splits."],
              ["3. EDA", "Profiled shape/types, missingness, duplicates, outliers, distributions, correlations, chronology and leakage/shortcut risks."],
              ["4. Preprocessing", "Imputed and encoded ML features; normalized images/sequences; created masks, velocities and train-only augmentations."],
              ["5. ML/DL", "Trained/compared LightGBM and XGBoost candidates; built ResNet-18 and BiLSTM paths."],
              ["6. Training", "Used group-aware CV for Stage 1 and validation-driven checkpointing for Stage 2."],
              ["7. Evaluation", "Tested frozen models on locked, patient-disjoint cohorts; reported metrics and errors."],
              ["8. Best model", "Stage 1 V4 regularized LightGBM was frozen; Stage 2 baseline checkpoints were preserved."],
              ["9. Integration", "Served frozen artifacts using FastAPI; Stage 2 added aggregation, fusion and dashboard/report outputs."],
              ["10. Final output", "Structured risk/probability outputs for research decision support, with a mandatory non-clinical disclaimer."],
          ], [4.2*cm, 11.9*cm]),
          P("Best 30-second complete answer", "Viva"),
          P("My project is a synthetic-data oncology decision-support prototype. In Stage 1 I cleaned encounter data, performed EDA, engineered leakage-safe features, and selected a regularized LightGBM model for Low, Moderate and High toxicity risk using patient-aware validation. I evaluated the frozen model on a locked test set. In Stage 2 I built a ResNet-18 pathology classifier and a BiLSTM for biomarker forecasting, protected the Day 0-90 prediction boundary, and integrated outputs through an API with missing-modality handling. The main limitation is that the data are synthetic, so it is not clinically validated.", "Viva"), PageBreak()]

qas = [
("What is your project?", "A synthetic-data oncology research decision-support prototype for toxicity-risk prediction, pathology classification and biomarker forecasting."),
("What problem did Stage 1 solve?", "It predicted Low, Moderate or High treatment toxicity risk from a patient encounter."),
("What type of ML task is that?", "Supervised multiclass classification."),
("What dataset did you use for Stage 1?", "A cleaned synthetic oncology dataset with 8,754 encounter rows from 6,000 patients."),
("What was the Stage 1 target?", "toxicity_risk with Low, Moderate and High classes."),
("What were the Stage 1 inputs?", "Clinical, treatment, genomic, biomarker and vital-sign fields; 30 raw features plus engineered features."),
("Why did you remove duplicates?", "Duplicate rows can overweight a patient pattern and distort evaluation. I removed 147 exact duplicates."),
("How did you handle missing values?", "Median imputation for numerical values, explicit Unknown/None categories for categorical/genomic values, and documented False imputation for a prior-adverse-event flag."),
("What feature engineering did you do?", "I created 11 clinical interaction/composite features, such as treatment load, organ impairment and vital instability, without using the target."),
("Why did you exclude patient_id?", "It is an identifier, not a clinical predictor; using it could let the model memorize patients."),
("Which ML models did you try?", "Tuned LightGBM, XGBoost with threshold adjustment, and multiple class-balance strategies."),
("Which ML model performed best?", "V4 regularized LightGBM was retained because it gave the best safety/generalization trade-off."),
("Why not select XGBoost V3?", "Its train-validation macro F1 gap was 0.2665, showing severe overfitting."),
("What is regularization in your project?", "Depth limits, L1/L2 penalties and minimum-child constraints that reduced model complexity and improved generalization."),
("How did you split the data?", "Patient-aware grouping: 80% training and a 20% locked test set, with group-aware cross-validation inside training."),
("Why split by patient?", "The same patient can have multiple encounters; row-only splitting would leak patient-specific patterns."),
("What is macro F1?", "It computes F1 for each class and averages them equally, so the majority Low-risk class does not dominate."),
("Why use high-risk recall?", "Missing a high-risk toxicity case is safety-critical, so I measured how many actual high-risk cases were found."),
("What was the final Stage 1 result?", "Locked-test macro F1 was 0.5288 and high-risk recall was 0.6287, with bootstrap confidence intervals."),
("What did error analysis show?", "Moderate was the hardest class; there were 50 High-to-Low critical misses, so the model requires clinician review."),
("Why use bootstrap confidence intervals?", "They show uncertainty around test metrics instead of presenting one estimate as exact."),
("Why did you use deep learning?", "Images and ordered biomarker sequences are better handled by CNN and sequence models than by a flat tabular model."),
("What did the CNN do?", "It classified 224 x 224 pathology tiles into benign, malignant or inflammation."),
("What CNN architecture did you implement?", "A pretrained ResNet-18 backbone with a 128-unit ReLU/dropout classification head; a custom CNN fallback also exists."),
("What image loss function did you use?", "Cross-entropy loss for the three-class pathology classifier."),
("What temporal model did you implement?", "A 2-layer bidirectional LSTM with 13 inputs and two output heads."),
("What did the temporal model predict?", "A 30-day ctDNA VAF forecast and a binary future progression trend."),
("What temporal features did you use?", "ctDNA, CEA, CA-125, LDH, CRP, backward-looking ctDNA velocity, delta days, days from baseline and five missingness masks."),
("Why include delta_days?", "Clinical visits are not equally spaced, so elapsed time helps the sequence model interpret change."),
("How did you avoid future leakage in DL?", "Inputs were restricted to Days 0-90; Day 91+ observations were targets only, and training statistics were used for normalization."),
("What DL losses did you use?", "Smooth L1 for ctDNA regression plus BCEWithLogits for progression classification."),
("What optimizer did you use?", "AdamW for both image and temporal baseline models."),
("What DL metrics did you use?", "Image accuracy/precision/recall/F1/ROC-AUC; temporal MAE/RMSE/R2 plus classification accuracy/precision/recall/F1/ROC-AUC/PR-AUC."),
("Why are the Stage 2 DL scores perfect?", "The audit attributes this to deterministic synthetic data separability. I report it as a technical benchmark, never as clinical performance."),
("What did the Data Engineer do?", "Built clean processed CSVs, validated data quality, generated Stage 2 modalities and enforced patient-level split manifests."),
("What did the EDA Engineer do?", "Checked data quality, distributions, missingness, class balance, leakage, chronology and image shortcut risks."),
("What did the Evaluation Engineer do?", "Evaluated frozen models on locked test data, computed metrics/intervals, and performed confusion-matrix and error analysis."),
("What did the Integration Engineer do?", "Connected frozen models to FastAPI endpoints and, in Stage 2, added aggregation, fusion and missing-modality behavior."),
("Is the project deployed clinically?", "No. It is a local research prototype with FastAPI and a Streamlit dashboard; it is not clinically validated or approved."),
("What should you say is the main limitation?", "All data are synthetic, so model scores demonstrate pipeline behavior only and cannot be used for medical decisions."),
]
story += [P("9. Forty Project-Specific Viva Questions", "H1x")]
for i, (q, a) in enumerate(qas, 1):
    story.append(P(f"<b>{i}. {q}</b><br/>{a}", "BodyX"))
    if i in (14, 27): story.append(PageBreak())
story.append(PageBreak())

story += [P("10. Final 5-Minute Revision Sheet", "H1x"),
          table(["Topic", "Remember this"], [
              ["Project problem", "Predict oncology toxicity risk and prototype pathology/biomarker signals using synthetic data."],
              ["Dataset", "Stage 1: 8,754 encounters/6,000 patients. Stage 2: 1,000 patients, 12,000 tiles, 16,012 observations."],
              ["Features", "Stage 1: clinical/treatment/genomic/biomarker/vitals plus 11 engineered features. Stage 2: image pixels or 13 temporal features."],
              ["Target", "Stage 1: Low/Moderate/High toxicity. Stage 2: tissue class; ctDNA 30-day value; progression trend."],
              ["Preprocessing", "Cleaning, imputation, encoding, training-only scaling/augmentation, missingness masks, leakage exclusion."],
              ["EDA", "Shape, types, missing values, duplicates, outliers, distributions, balance, correlation, chronology, leakage and image shortcuts."],
              ["ML model", "V4 regularized LightGBM; selected over V3 due to stronger generalization and high-risk recall."],
              ["DL models", "ResNet-18 pathology classifier; 2-layer BiLSTM multi-task forecaster."],
              ["Evaluation", "Locked patient-disjoint test sets, macro F1, high-risk recall, accuracy, precision/recall, MAE/RMSE/R2, ROC-AUC/PR-AUC."],
              ["Best Stage 1 result", "V4 test macro F1 0.5288; high-risk recall 0.6287; accuracy 0.5766."],
              ["Integration", "FastAPI for frozen inference; Stage 2 has aggregation/fusion and missing-modality statuses."],
              ["Final output", "Structured research risk/probability outputs with a mandatory non-clinical disclaimer."],
          ], [4.2*cm, 11.9*cm]),
          P("20 things to remember", "H2x"),
          table(["#", "Remember"], [[str(i+1), x] for i,x in enumerate([
              "Synthetic data means no clinical claim.", "Stage 1 target is three-class toxicity risk.", "Stage 1 final model is V4 regularized LightGBM.", "V4 was selected for generalization, not just one score.", "Patient-level split prevents encounter leakage.", "Macro F1 was the primary Stage 1 metric.", "High-risk recall was safety-critical.", "V4 test macro F1 = 0.5288.", "V4 high-risk recall = 0.6287.", "50 High-to-Low test errors were critical misses.", "Stage 2 pathology uses ResNet-18 transfer learning.", "Stage 2 temporal model is a 2-layer BiLSTM.", "Temporal inputs are only Day 0-90.", "Future targets are not model inputs.", "Missing temporal data uses masks plus imputation.", "Image augmentation is training-only.", "The API serves frozen models; it does not retrain.", "Stage 2 supports partial modalities explicitly.", "DL perfect synthetic metrics are not clinical evidence.", "Always finish by stating the research-only limitation."
          ])], [1.2*cm, 14.9*cm]),
          Spacer(1, .3*cm), P("Final answer to memorize", "H2x"),
          P("I built the whole pipeline: clean data, EDA, leakage-safe modeling, independent evaluation, and API integration. The strongest completed ML model is V4 regularized LightGBM. Stage 2 extends the project to pathology images and biomarker sequences using ResNet-18 and BiLSTM. The work is technically implemented and evaluated on synthetic data, but it is a research prototype and not a clinical decision system.", "Viva")]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=1.55*cm, leftMargin=1.55*cm, topMargin=1.45*cm, bottomMargin=1.7*cm)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
