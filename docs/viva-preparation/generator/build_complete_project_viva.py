from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, KeepTogether

OUT = Path(r"C:\Users\nisham\Desktop\ONCOLOGY TREATMENT\output\pdf\complete_project_specific_data_science_viva.pdf")
OUT.parent.mkdir(parents=True, exist_ok=True)
BLUE=colors.HexColor('#1F2E85'); RED=colors.HexColor('#D9322B'); DARK=colors.HexColor('#34464A'); LIGHT=colors.HexColor('#EAF0F1'); INK=colors.HexColor('#182126')
ss=getSampleStyleSheet()
ss.add(ParagraphStyle(name='Cover',fontName='Helvetica-Bold',fontSize=19,leading=24,textColor=BLUE,alignment=TA_CENTER))
ss.add(ParagraphStyle(name='Cover2',fontName='Helvetica',fontSize=9,leading=13,textColor=DARK,alignment=TA_CENTER))
ss.add(ParagraphStyle(name='Part',fontName='Helvetica-Bold',fontSize=13,leading=16,textColor=colors.white))
ss.add(ParagraphStyle(name='Sec',fontName='Helvetica-Bold',fontSize=10.5,leading=13,textColor=colors.HexColor('#1765B1'),spaceBefore=5,spaceAfter=4))
ss.add(ParagraphStyle(name='Q',fontName='Helvetica-Bold',fontSize=7.6,leading=9.7,textColor=RED,spaceBefore=3,spaceAfter=1))
ss.add(ParagraphStyle(name='A',fontName='Helvetica',fontSize=7.45,leading=9.6,textColor=INK,leftIndent=4,spaceAfter=3))
ss.add(ParagraphStyle(name='Small',fontName='Helvetica',fontSize=6.6,leading=8.2,textColor=INK))
ss.add(ParagraphStyle(name='Call',fontName='Helvetica-Bold',fontSize=7.6,leading=10,textColor=DARK))

def P(x,s='A'): return Paragraph(x,ss[s])
def qa(q,a): return [P('Q: '+q,'Q'),P('A: '+a,'A')]
def add_qas(story,items):
    for q,a in items: story.extend(qa(q,a))
def bar(x):
    t=Table([[P(x,'Part')]],colWidths=[16.6*cm]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),BLUE),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)])); return t
def box(x):
    t=Table([[P(x,'Call')]],colWidths=[16.6*cm]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),LIGHT),('BOX',(0,0),(-1,-1),.3,colors.HexColor('#B7C8CB')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)])); return t
def sec(story,title): story.append(P(title,'Sec'))
def page(story,title): story += [PageBreak(),bar(title)]
def footer(canvas,doc):
    canvas.saveState(); canvas.setStrokeColor(colors.HexColor('#9AA8AD')); canvas.line(1.6*cm,1.15*cm,19.4*cm,1.15*cm); canvas.setFillColor(colors.HexColor('#6D7B80')); canvas.setFont('Helvetica',5.3); canvas.drawString(1.6*cm,.82*cm,'Project-Specific Data Science Viva | Oncology Treatment Optimization'); canvas.drawRightString(19.4*cm,.82*cm,f'Page {doc.page}'); canvas.restoreState()

story=[Spacer(1,3.2*cm),P('Complete Project-Specific<br/>Data Science Viva Preparation','Cover'),Spacer(1,.35*cm),P('Personalized Precision Medicine for Oncology Treatment Optimization','Cover2'),Spacer(1,.7*cm),Table([['']],colWidths=[11*cm],rowHeights=[.03*cm],style=[('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#A5BAC0'))]),Spacer(1,.8*cm),P('Faculty-style WHY, HOW, WHAT, WHICH, WHEN and WHAT-IF questions based only on the implemented project.','Cover2'),Spacer(1,1.8*cm),box('Scientific honesty: Stage 1 metrics are locked-test ML results. Stage 2 results come from a deterministic synthetic benchmark and do not establish clinical performance.'),PageBreak(),bar('PART 1: COMPLETE PROJECT UNDERSTANDING')]

add_qas(story,[
('What is the complete project problem?','The project builds research prototypes for oncology treatment optimization. Stage 1 predicts treatment-toxicity risk as Low, Moderate or High. Stage 2 predicts pathology class, progression risk, and 30-day ctDNA from synthetic multimodal data.'),
('What is the main objective?','To process oncology data safely, train patient-level predictive models, evaluate them honestly, and expose predictions through application interfaces without claiming clinical validity.'),
('What is the Stage 1 dataset?','The raw oncology CSV had 8,901 rows and 36 columns. The cleaned master dataset has 8,754 encounters, 35 columns, and 6,000 unique patients.'),
('What is the Stage 2 dataset?','A controlled synthetic NSCLC benchmark with 1,000 patients, 12,000 pathology tiles, and 16,012 longitudinal biomarker observations.'),
('What are the targets?','Stage 1: toxicity_risk with Low, Moderate and High classes. Stage 2 pathology: benign, malignant or inflammation. Stage 2 temporal: progression/recurrence classification and a 30-day ctDNA regression forecast.'),
('What are the best validated models?','Stage 1 selected V4 regularized LightGBM. The validated Stage 2 baselines are pretrained ResNet-18 with a frozen backbone and a two-layer bidirectional LSTM. Upgrade modules must not be described as winning without completed experimental evidence.'),
('What is the complete data flow?','Raw data -> validation and cleaning -> cleaned dataset -> feature engineering and model-ready transformation -> patient-safe training -> validation-based selection -> locked-test evaluation -> saved model/checkpoint -> API or dashboard -> prediction output.'),
('What backend, frontend and database were implemented?','FastAPI is the backend. Stage 2 also has a Streamlit dashboard and standalone report. No database was implemented; data and artifacts are stored as CSV, JSON, joblib files and PyTorch checkpoints.'),
('Was the project deployed clinically?','No. It is a research prototype. Production and clinical deployment are not specified as completed in the project.'),
])

page(story,'PART 2: RAW DATA vs CLEANED DATA vs FEATURE DATA')
sec(story,'The difference faculty expects you to explain')
tbl=Table([
    [P('<b>Raw Data</b>','Small'),P('<b>Cleaned Data</b>','Small'),P('<b>Feature Data</b>','Small')],
    [P('Data as originally collected, before correction. Stage 1: oncology_uncleaned.csv with 8,901 rows, inconsistent missing tokens, duplicates, units, category variants and invalid values.','Small'),P('Validated and standardized records. Stage 1: master_patient_dataset.csv with 8,754 rows after 147 duplicates were removed and missing/invalid values were handled.','Small'),P('The exact model inputs after selecting raw predictors, excluding leakage/IDs, creating 11 engineered variables, encoding categories and applying the fitted preprocessing pipeline.','Small')],
    [P('Suitable for audit, not direct training.','Small'),P('Consistent and analyzable, but may still contain IDs, target, unnecessary fields or unencoded categories.','Small'),P('Numerical representation consumed by the model; transformations are learned from training data only.','Small')]
],colWidths=[5.53*cm]*3)
tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),BLUE),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#B9C7CA')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('BACKGROUND',(0,1),(-1,-1),colors.HexColor('#F4F7F8'))])); story.append(tbl)
add_qas(story,[
('Give the shortest difference.','Raw data is what we received. Cleaned data is corrected and standardized. Feature data is the selected and transformed input given to the model.'),
('Why could you not train directly on raw Stage 1 data?','It contained duplicate records, multiple missing-value symbols, unit text inside numeric fields, category variations and implausible values. These would create inconsistent inputs and biased evaluation.'),
('Is cleaned data automatically ML-ready?','No. Cleaned data can still contain IDs, the target, leakage columns, categorical text and unnecessary variables. It still needs feature selection, engineering and model-specific transformation.'),
('What is the difference between cleaning and feature engineering?','Cleaning repairs data quality. Feature engineering creates useful model signals, such as cumulative treatment load, organ impairment or biomarker severity.'),
('Can cleaned data contain unnecessary features?','Yes. patient_id is useful for grouping but should not become a predictor. treatment_response was excluded because it could leak outcome information.'),
('Which stage creates ML-ready input?','The feature and preprocessing pipeline: predictor selection, engineered features, categorical encoding and other fitted transformations convert cleaned records into the model matrix.'),
('What happens if an important feature is removed?','The model may lose signal and recall or F1 may fall. The correct response is to retrain and compare on validation data, not assume the effect.'),
('What happens if irrelevant features are included?','They can add noise, increase dimensionality and overfitting, and make explanations less stable. Validation determines whether they help.'),
('What is the relationship between features and target?','Features are information available at prediction time; the target is the outcome to learn. A feature must not directly or indirectly reveal the target.'),
])

page(story,'PART 3: WHY DID YOU DO THIS? - DECISION DEFENCE')
add_qas(story,[
('Why clean the data?','Because the model learns whatever patterns exist, including errors. Cleaning made representations consistent before analysis and training.'),
('Why remove duplicates?','Repeated encounters could receive extra influence and could make performance look better. The pipeline removed 147 duplicate rows.'),
('Why use median for missing numeric values?','The implemented Stage 1 pipeline used median imputation because it is less affected by extreme values than the mean. The value was learned within training folds to avoid leakage.'),
('Why use Unknown for missing categories?','It preserves the record and represents missing categorical information explicitly rather than inventing an existing category.'),
('Why convert invalid values to missing first?','Values outside plausible limits should not be treated as real measurements. The pipeline converted 11 ages, 12 heart rates, 17 systolic pressures and 12 WBC values to missing before imputation.'),
('Why encode categorical variables?','LightGBM received a numeric model matrix, so category text had to be converted consistently. Unknown categories at inference are safely ignored by the fitted encoder.'),
('Why create engineered features?','Single measurements may not express combined clinical burden. The 11 engineered features summarize treatment load, organ impairment, vital instability, genomic instability and biomarker severity.'),
('Why compare LightGBM and XGBoost?','A comparison tests whether the selected result depends on one algorithm. The project chose using validation behavior and generalization, not model reputation.'),
('Why choose V4 LightGBM?','It offered a useful balance of macro F1, High-risk recall and stability with a much smaller train-to-validation gap than the rejected XGBoost candidate.'),
('Why did you not choose the model with the highest training score?','Training score measures memorization of seen examples. The XGBoost candidate had train macro F1 0.7763 but CV macro F1 0.5427, a severe gap.'),
('Why use patient-grouped cross-validation?','One patient may have multiple encounters. Grouping prevents the same patient appearing in training and validation, which would overestimate generalization.'),
('Why compare class weighting and resampling?','The High-risk class was only 19.0%. The project tested whether rebalancing could improve minority-class performance without damaging stability.'),
('Why not force SMOTE?','In this project it distorted the 113-dimensional mix of one-hot and continuous values and did not produce the preferred recall/stability tradeoff.'),
('Why use ResNet-18 for pathology?','The implemented baseline uses pretrained image features and a small trainable head, which is practical for the synthetic 224 x 224 tile benchmark.'),
('Why use a BiLSTM for temporal data?','The prediction depends on the order and evolution of biomarkers. The two-layer bidirectional LSTM models patterns across the available historical window.'),
('Why use AdamW?','Both Stage 2 baselines used AdamW because it combines adaptive optimization with explicit weight decay. This answer is based on the implemented configuration.'),
('Why use cross-entropy for pathology?','The pathology task has three mutually exclusive classes, so cross-entropy trains the model to assign probability to the correct class.'),
('Why use Smooth L1 plus BCEWithLogits?','The temporal model has two tasks: Smooth L1 for ctDNA regression and BCEWithLogits for binary progression classification.'),
('Why use fixed fusion?','The implemented prototype combines three available risk signals using documented fixed weights. It is transparent, but the weights are not clinically validated.'),
])

page(story,'PART 4: METRIC DEFENCE - ACCURACY, PRECISION, RECALL, F1')
sec(story,'Strong answer: Why focus more on Recall than Accuracy?')
story.append(box('In Stage 1, the High-risk class represents patients the model flags for greater toxicity concern. High-risk Recall = TP / (TP + FN): it asks how many actual High-risk cases we detected. A false negative means an actual High-risk case was predicted as a lower class. Accuracy can look acceptable because Low risk is 53.4% of the data, even while important High-risk cases are missed. Therefore, we examine High-risk recall alongside macro F1, precision and the confusion matrix. We do not use Recall alone because predicting too many cases as High would increase false positives and reduce precision.'))
add_qas(story,[
('What is Accuracy?','Correct predictions divided by all predictions. Stage 1 locked-test accuracy was 0.5766. Its limitation is sensitivity to class imbalance.'),
('What is Precision?','TP / (TP + FP). For a selected class, it asks: when the model predicts this class, how often is it correct?'),
('What is Recall?','TP / (TP + FN). For a selected class, it asks: of all actual cases in this class, how many did the model find?'),
('What is F1-score?','2 x Precision x Recall / (Precision + Recall). It rewards a balance; a model cannot get high F1 if either precision or recall is very poor.'),
('What is macro F1?','Compute F1 for each class, then average them equally. It is suitable because Low, Moderate and High should each affect model selection.'),
('What is a Stage 1 false negative example?','For High-risk detection, an actual High-risk encounter predicted Moderate or Low is a false negative. The most serious observed transition was High to Low: 50 cases.'),
('What is a Stage 1 false positive example?','An actual Low or Moderate encounter predicted High is a High-risk false positive. It could cause unnecessary escalation or review.'),
('Can Accuracy be high while Recall is low?','Yes. With an imbalanced dataset, a model can correctly predict the majority class often while missing many minority High-risk cases.'),
('Why not use Accuracy alone?','It compresses all errors into one number and does not show which class was missed. Our error costs are not identical.'),
('Why not use Recall alone?','A model could label almost everyone High risk and obtain high High-risk recall while producing many false alarms. Precision and macro F1 reveal that problem.'),
('When would Precision be prioritized?','When false alarms cause substantial harm or cost. In this project, precision matters because incorrect High-risk alerts can trigger unnecessary attention.'),
('When would Recall be prioritized?','When missing actual positive cases is especially concerning. That is why Stage 1 explicitly reports High-risk recall.'),
('What happens if Recall decreases?','More actual High-risk cases are missed. We inspect false negatives and the confusion matrix before deciding whether a threshold or model change is justified.'),
('What happens if Precision decreases?','More predicted High-risk cases are false alarms, increasing unnecessary review and reducing trust.'),
('What did the locked test show?','Macro F1 0.5288, High-risk recall 0.6287, accuracy 0.5766, macro precision 0.5209, macro recall 0.5440, and mean Brier score 0.1817.'),
])

page(story,'PART 4: METRIC DEFENCE - OTHER ACTUAL METRICS')
add_qas(story,[
('What is ROC-AUC?','It measures ranking across classification thresholds. Stage 2 reported it for image and progression classification; it should be read with PR-AUC and class metrics.'),
('What is PR-AUC?','It summarizes precision-recall performance across thresholds and is useful when the positive class is important or imbalanced.'),
('What is log loss?','It penalizes incorrect probabilities, especially confident wrong predictions. Stage 1 locked-test log loss was 0.9129.'),
('What is Brier score?','The mean squared difference between predicted probability and actual outcome indicator. Lower is better; Stage 1 mean Brier score was 0.1817.'),
('What is MAE?','Mean absolute error: the average absolute forecast error in ctDNA units. Stage 2 test MAE was 0.3698.'),
('What is RMSE?','The square root of mean squared error. It penalizes larger ctDNA errors more strongly; Stage 2 test RMSE was 0.5060.'),
('What is R2?','It measures how much target variation is explained relative to predicting the mean. Stage 2 test R2 was 0.8236 on the synthetic benchmark.'),
('Why report confidence intervals?','A single test score varies with the sample. Stage 1 bootstrap CI for macro F1 was [0.5035, 0.5521] and for High-risk recall [0.5759, 0.6783].'),
('How do you know the model is actually performing well?','We do not rely on one number. We check patient-safe validation, locked-test metrics, class-level results, confusion matrices, confidence intervals, calibration/error evidence and limitations.'),
('Does a high metric prove clinical usefulness?','No. Stage 2 produced perfect classification on synthetic data because the generator is highly separable. Clinical usefulness requires external real-world validation.'),
])

page(story,'PART 5: MODEL SELECTION DEFENCE')
add_qas(story,[
('What is LightGBM in your project?','A gradient-boosted tree classifier for Stage 1 tabular data. It models nonlinear interactions and handles complex feature relationships efficiently.'),
('What are its strengths here?','Strong tabular performance, manageable training cost, regularization controls and feature-importance support.'),
('What are its weaknesses?','It can overfit, probabilities may require calibration, and feature importance does not prove causation.'),
('What is XGBoost in your project?','A second gradient-boosted tree candidate used for comparison. Its selected candidate overfit more strongly than V4 LightGBM.'),
('Why did XGBoost perform worse for selection?','Not simply because XGBoost is worse. Under the tested configuration, its train-to-CV macro-F1 gap was 0.2665, indicating weak generalization.'),
('What is ResNet-18 in your project?','A pretrained convolutional neural network baseline for three-class pathology-tile classification, with a frozen backbone and trainable classification head.'),
('What is the BiLSTM in your project?','A two-layer bidirectional sequence model with 13 input features, hidden size 64 per direction and separate regression/classification heads.'),
('Which model is most interpretable?','The Stage 1 tree model is easier to inspect with feature importance than the DL models, although importance still does not prove causal effect.'),
('Which is more computationally expensive?','Image CNN training is generally heavier than the Stage 1 tree pipeline. Exact cross-model hardware-normalized cost comparison is not specified in the project.'),
('What if the dataset becomes larger?','Retrain and re-evaluate. Tree and neural models may gain stability, but storage, training time and monitoring costs increase. More rows do not help if quality or representativeness is poor.'),
('Which would you choose for production?','For Stage 1 tabular toxicity prediction, V4 LightGBM is the validated candidate. I would still require external validation and governance before any real production use.'),
('Did ResNet-50 beat ResNet-18?','Not established by the validated results used in this guide. I would not claim ResNet-50 wins until both are trained and evaluated fairly.'),
])

page(story,'PART 6: PREPROCESSING DEFENCE - WHAT, WHY, HOW, WHAT IF')
rows=[
('Missing numeric values','Median imputation','Keep records and use a robust central value','Applied in the preprocessing pipeline','Unresolved NaNs could break training or bias row removal'),
('Missing categories','Unknown or documented defaults','Preserve missingness without inventing a real label','Standardized tokens before encoding','Inconsistent categories or lost rows'),
('Duplicates','Removed 147 duplicate rows','Prevent repeated samples receiving extra weight','Deduplication in cleaning pipeline','Biased learning and optimistic evaluation'),
('Invalid ranges','Convert implausible values to missing','Do not learn from impossible measurements','Bounds checks, then imputation','Spurious relationships and unstable predictions'),
('Units and text','Strip/standardize units','Create consistent numeric fields','Parsed age, ctDNA, oxygen and drug-dose strings','Same quantity may be treated as different values'),
('Categorical encoding','One-hot style preprocessing','Convert text categories to model input','Fitted inside training folds','Model cannot consume inconsistent strings; leakage if fit globally'),
('Feature engineering','Create 11 combined variables','Represent clinically motivated burden/instability signals','Deterministic transformations inside pipeline','May lose useful combined patterns'),
('Temporal scaling','Training-only standardization','Put biomarker scales on comparable numerical ranges','Fit scaler on train patients; reuse for val/test','Optimization instability or leakage if fitted on test'),
('Image normalization','ImageNet normalization','Match pretrained ResNet input convention','Applied after resizing to 224 x 224','Transfer features may receive unexpected input scale'),
('Data splitting','Patient-disjoint partitions','Measure new-patient generalization','Grouped folds and 700/150/150 Stage 2 split','Same-patient leakage and inflated metrics'),
]
t=Table([[P('<b>Operation</b>','Small'),P('<b>What</b>','Small'),P('<b>Why</b>','Small'),P('<b>How</b>','Small'),P('<b>If omitted</b>','Small')]]+[[P(c,'Small') for c in r] for r in rows],colWidths=[2.5*cm,3.0*cm,3.8*cm,3.6*cm,3.7*cm],repeatRows=1)
t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),BLUE),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.25,colors.HexColor('#BCC8CB')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F3F7F7')])]))
story += [t,Spacer(1,.2*cm),box('Important defence: preprocessing can leak data. Imputation, scaling, encoding and resampling must be fitted using training data only, then reused unchanged on validation and test data.')]

page(story,'PART 7: EDA DEFENCE')
add_qas(story,[
('Why perform EDA?','To understand structure, missingness, invalid values, class balance, distributions, relationships, time order and possible visual shortcuts before choosing preprocessing and evaluation.'),
('What did Stage 1 EDA discover?','8,754 cleaned rows, 35 columns, class imbalance, missing/duplicate problems in raw data, distributions, correlations and feature relationships.'),
('What did the target distribution change?','Low was 53.4%, Moderate 27.6%, High 19.0%. This motivated macro F1, High-risk recall and balancing experiments.'),
('What do distribution plots show?','Range, skew, concentration and possible outliers. They helped validate cleaning and understand biomarker/vital ranges.'),
('Why use a correlation heatmap?','To inspect linear relationships and redundancy among numerical variables. Correlation supported exploration, not causal claims.'),
('Did EDA influence modeling?','Yes. Class balance influenced metrics and weighting; quality findings drove cleaning; temporal checks drove masks/time features; image audits checked shortcut risks.'),
('What did pathology EDA show?','12,000 readable 224 x 224 RGB images, exact class balance, no blank/corrupt tiles, and randomized appearance factors intended to reduce class shortcuts.'),
('What did temporal EDA show?','16,012 observations, no temporal inversions, roughly 5.31%-5.60% missingness, and zero violations of the Day 0-90 input cutoff.'),
('What if EDA were skipped?','We could train on duplicates, impossible values, leakage, mislabeled time windows or shortcut features and misinterpret apparently strong scores.'),
('What was the most important EDA warning?','The Stage 2 synthetic generator creates separable patterns; perfect classification does not imply clinical performance.'),
])

page(story,'PART 8: TRAINING, VALIDATION, TESTING AND LEAKAGE')
add_qas(story,[
('Why not train and test on the same data?','The model has already seen the examples, so the score measures memorization rather than new-patient generalization.'),
('What is training data?','Data used to fit model parameters and preprocessing statistics.'),
('What is validation data?','Unseen development data used to compare models, thresholds and hyperparameters.'),
('What is test data?','A locked final dataset used after selection to estimate unbiased performance.'),
('What cross-validation was used?','Five-fold StratifiedGroupKFold grouped by patient_id for Stage 1 model selection.'),
('How were Stage 2 patients split?','700 train, 150 validation and 150 test patients, with zero overlap.'),
('What is data leakage?','Information unavailable at real prediction time, or information from validation/test, enters model training or preprocessing.'),
('How was Stage 1 leakage avoided?','Patient grouping, fold-local preprocessing/resampling, exclusion of treatment_response and IDs from predictors, and a locked test set.'),
('How was Stage 2 leakage avoided?','Day 0-90 was input and Day 91+ target, patients were disjoint, target/trajectory fields were excluded, and temporal scalers were fitted only on training patients.'),
('Why can training performance be high but test performance low?','The model may memorize noise or dataset-specific patterns. The rejected XGBoost candidate demonstrated a large train-to-CV gap.'),
('How do you detect overfitting?','Compare training and validation trajectories/scores, monitor the gap, and confirm on patient-disjoint data.'),
('How do you know the model generalizes?','Only within the tested distribution: patient-disjoint validation, stable folds and locked test results. External clinical generalization is still unproven.'),
])

page(story,'PART 9: DEEP LEARNING DEFENCE')
add_qas(story,[
('Why use deep learning in Stage 2?','Pathology images and longitudinal sequences have spatial and temporal structure that CNNs and sequence models are designed to learn.'),
('Why not use traditional ML for images directly?','Raw pixels are high-dimensional and spatially organized. The CNN learns hierarchical visual representations. A fair alternative using engineered image features was not reported.'),
('Explain the pathology architecture.','A pretrained ResNet-18 backbone produces a 512-dimensional representation. A custom 128-unit ReLU/dropout head outputs three pathology logits.'),
('Explain the temporal architecture.','Thirteen features enter a two-layer bidirectional LSTM. Its 128-dimensional representation feeds a ctDNA regression head and a progression binary-classification head.'),
('Why ReLU?','It is the implemented nonlinear activation in the custom heads and allows the networks to learn nonlinear mappings.'),
('What is dropout?','During training it randomly suppresses units to reduce co-adaptation. Pathology head dropout is 0.3; temporal heads use dropout in the implemented architecture.'),
('What is an epoch?','One pass through the training set. Pathology used 3 epochs; temporal used 25.'),
('What is a batch?','A subset processed before an optimizer update. Pathology batch size was 64; temporal batch size 32.'),
('What is backpropagation?','The method that calculates how each trainable weight affected the loss, enabling gradient-based updates.'),
('How does the network learn?','Forward pass -> compute loss -> backpropagate gradients -> AdamW update -> repeat across batches and epochs.'),
('How was the final DL model evaluated?','On 1,800 held-out tiles and 150 held-out patients, with classification and ctDNA regression metrics plus diagnostic explainability and robustness checks.'),
('Why are perfect Stage 2 classification metrics suspicious?','Because real medical data are rarely perfectly separable. The project correctly attributes them to the deterministic synthetic benchmark and avoids clinical claims.'),
])

page(story,'PART 10: ENGINEERING ROLE QUESTIONS')
add_qas(story,[
('What did you do as Data Engineer?','Created reproducible cleaning/validation pipelines, CSV master data, synthetic multimodal datasets and patient split manifests.'),
('Where is data stored?','In CSV datasets and image folders; trained artifacts are joblib, JSON and .pt checkpoints. No database is implemented.'),
('What did you do as EDA Engineer?','Audited structure, quality, class balance, distributions, correlations, chronology, missingness and image shortcut risks.'),
('What did you do as ML Engineer?','Built leakage-safe features, trained LightGBM/XGBoost candidates, selected V4 through patient-grouped validation, froze artifacts and implemented inference.'),
('How is the Stage 1 model saved and loaded?','The selected model and preprocessor are serialized as joblib artifacts, with target mapping/configuration stored separately for consistent inference.'),
('What did you do as DL Engineer?','Implemented the ResNet-18 tile classifier, BiLSTM multi-task model, transforms, sequence handling, checkpointing and patient-level prediction flow.'),
('What did you do as Evaluation Engineer?','Used locked tests, class/regression metrics, bootstrap intervals, confusion matrices, error transitions, calibration-related scores and robustness diagnostics.'),
('What did you do as Integration Engineer?','Connected validation schemas, preprocessing, frozen models and response formatting through FastAPI; Stage 2 also connects inference to a Streamlit dashboard/report.'),
('How does a Stage 1 API request flow?','JSON request -> Pydantic validation -> feature engineering -> frozen preprocessor -> LightGBM probabilities -> decision logic -> risk class and probabilities.'),
('How does Stage 2 missing-modality handling work?','The response declares FULL_MULTIMODAL, PATHOLOGY_ONLY, TEMPORAL_ONLY or INSUFFICIENT_DATA and uses the available prediction route.'),
])

page(story,'PART 11: PROJECT-SPECIFIC WHAT-IF QUESTIONS')
add_qas(story,[
('What if missingness increases?','First identify whether it is random or systematic. Re-run data-quality and subgroup evaluation, retrain the imputation/mask-aware pipeline and check calibration and recall.'),
('What if class imbalance becomes stronger?','Accuracy becomes less informative. Re-evaluate class weights/resampling within CV and monitor per-class precision, recall, macro F1 and PR-AUC.'),
('What if a feature is removed?','Retrain the complete pipeline and compare validation/test behavior. Do not infer importance only from one feature-importance chart.'),
('What if an irrelevant feature is added?','It may add noise or overfitting. Keep it only if patient-safe validation shows consistent benefit and it is available at inference.'),
('What if Recall is low but Accuracy is high?','The model may be favoring the majority class and missing High-risk patients. Inspect the confusion matrix, per-class thresholds, weights and training data.'),
('What if the model overfits?','Increase regularization, simplify the model, improve data, adjust early stopping and re-evaluate with patient-grouped validation.'),
('What if locked-test performance is poor?','Report it honestly. Return to training/validation for changes, then preserve the test boundary; do not tune repeatedly on the test set.'),
('What if a new category appears in the API?','The Stage 1 encoder uses handle_unknown=ignore, so inference continues, but drift monitoring should record the unseen category.'),
('What if the input format changes?','The schema may reject it or preprocessing may fail. Version the API/schema, validate conversion and retrain if feature meaning changes.'),
('What if the model gives an incorrect prediction?','Store inputs/output for audit where permitted, review confidence and error category, communicate prototype limits, and never let the model replace clinical judgment.'),
('What if the dataset becomes ten times larger?','Use scalable storage/loading and retrain. Larger data may enable stronger models, but quality, leakage and representativeness still matter.'),
('What if one Stage 2 modality is missing?','Use the explicit pathology-only or temporal-only route; if neither is adequate, return INSUFFICIENT_DATA rather than fabricate a full prediction.'),
('What if visits are irregular?','Use days_from_baseline and delta_days. Do not treat visit positions as equally spaced.'),
('What if the data distribution changes after deployment?','Monitor input distributions, missingness, performance and calibration; investigate drift and retrain only with governed, newly validated data.'),
('What if ResNet-50 performs worse than ResNet-18?','Report the result and retain the better validated model. Architecture size does not guarantee better generalization.'),
])

chains=[
('Raw vs cleaned',['Raw is the original inconsistent input; cleaned is corrected and standardized.','No. Cleaned data may still include IDs, target/leakage columns and categorical text.','Feature engineering creates model signals; preprocessing converts them into the fitted model representation.']),
('Why Recall',['High-risk recall measures the fraction of actual High-risk cases detected.','An actual High-risk case predicted as Moderate or Low.','Because Recall alone can create too many false alarms; I also use precision, macro F1 and confusion matrices.']),
('Why not Accuracy',['The majority Low class can make Accuracy look acceptable while High-risk cases are missed.','Yes; predicting the majority class can produce this pattern.','Per-class precision/recall/F1, macro F1 and the confusion matrix.']),
('Patient split',['The same patient can have multiple encounters or tiles.','The model could recognize patient-specific patterns and inflate validation results.','Stage 1 grouped CV; Stage 2 700/150/150 patient-disjoint splits.']),
('V4 selection',['Regularized LightGBM V4.','Balanced macro F1 and High-risk recall with a manageable generalization gap.','Its tested configuration had a severe train-to-CV gap of 0.2665.']),
('SMOTE',['Synthetic minority oversampling.','The tested mixed 113-dimensional representation was distorted and did not improve the preferred stability/recall tradeoff.','Balanced class weights in V4, after comparing multiple strategies.']),
('Missing data',['Numeric median; categorical Unknown/documented defaults.','It keeps data usable and is robust to extreme numeric values.','Inside each training fold, then applied unchanged to validation/test.']),
('Feature leakage',['A feature contains information unavailable at prediction time.','treatment_response was excluded; IDs were not predictors.','Performance becomes falsely optimistic and fails in real use.']),
('Stage 2 cutoff',['Day 0-90 is input; Day 91+ provides forecasting targets.','Using future observations would reveal the answer.','Automated boundary checks and patient split checks.']),
('Perfect DL results',['Yes, all Stage 2 classification metrics were 1.0 on the held-out synthetic benchmark.','No. Deterministic synthetic patterns were unusually separable.','Only as evidence that the pipeline learned the benchmark; not clinical effectiveness.']),
('ResNet-18',['The validated pathology baseline.','Pretrained backbone, frozen initially, and trainable 512-to-128-to-3 head.','ResNet-50 is an upgrade target, but superiority is not established here.']),
('BiLSTM',['A two-layer bidirectional temporal model with dual heads.','It uses ordered historical biomarkers to forecast ctDNA and classify progression.','Biomarkers, velocity, days_from_baseline, delta_days and missingness masks.']),
('Multi-task loss',['Smooth L1 plus BCEWithLogits.','Smooth L1 trains ctDNA regression; BCEWithLogits trains binary progression.','Lambda 1 in the implemented setup.']),
('Fusion',['Fixed weighted fusion of pathology, progression and normalized ctDNA.','0.35, 0.40 and 0.25 respectively.','No. It is transparent prototype logic and not clinically validated.']),
('Calibration',['Whether predicted probabilities match observed frequencies.','Stage 1 reports Brier and log loss; explicit full calibration proof is limited.','Do not describe probabilities as reliable clinical risk without further validation.']),
('EDA',['To find quality problems, imbalance, temporal leakage and shortcuts before modeling.','The High-risk class is only 19.0%.','It drove macro F1/High-risk recall and balancing experiments.']),
('Overfitting',['Large training-to-validation performance gap.','Rejected XGBoost train macro F1 0.7763 vs CV 0.5427.','Regularization, simpler settings and validation-based selection.']),
('API',['Request validation, preprocessing, prediction and response formatting.','/health, /predict and /predict/batch in Stage 1.','Risk class plus class probabilities.']),
('No database',['The implementation stores CSV data and serialized artifacts.','No database was required for the research pipeline.','A governed database could be added for production; it is not currently implemented.']),
('Model failure',['Review false positives, false negatives, confidence and affected subgroup.','High to Low is the most concerning Stage 1 transition; 50 occurred.','Report it, investigate inputs and retrain only through the validation protocol.']),
('More data',['Not automatically.','It must be representative, correctly labeled and leakage-free.','More biased or duplicated data can worsen the system.']),
('Clinical use',['No.','Synthetic Stage 2 data and limited validation.','External cohorts, calibration, fairness, governance, security, monitoring and clinician-led trials.']),
]
page(story,'PART 12: CROSS-QUESTIONING CHAINS (1-11)')
for i,(topic,answers) in enumerate(chains[:11],1):
    block=[P(f'{i}. Faculty chain: {topic}','Sec')]
    prompts=['Give your first answer.','Why does that matter in this project?','What is the important limitation or follow-up?']
    for j,(q,a) in enumerate(zip(prompts,answers),1): block += qa(q,a)
    story.append(KeepTogether(block))
page(story,'PART 12: CROSS-QUESTIONING CHAINS (12-22)')
for i,(topic,answers) in enumerate(chains[11:],12):
    block=[P(f'{i}. Faculty chain: {topic}','Sec')]
    prompts=['Give your first answer.','Why or how was it used?','What is the important limitation or follow-up?']
    for q,a in zip(prompts,answers): block += qa(q,a)
    story.append(KeepTogether(block))

page(story,'PART 13: TRICK QUESTIONS')
add_qas(story,[
('Is cleaned data the same as feature data?','No. Cleaning repairs records; feature data is selected, engineered and transformed for a model.'),
('Is higher Accuracy always better?','No. It can hide weak minority-class recall and does not describe error cost.'),
('Is 100% training Accuracy good?','Not necessarily. It may indicate memorization. Validation/test evidence is required.'),
('Can preprocessing introduce leakage?','Yes, if imputation, scaling, encoding or resampling learns from validation/test data.'),
('Is more data always better?','No. Quality, representativeness, labels and independence matter.'),
('Is a complex model always better?','No. The larger model may overfit or add cost without validation improvement.'),
('Does correlation prove causation?','No. It only describes association.'),
('Can we evaluate only using Accuracy?','No, especially with Stage 1 imbalance. Use class metrics, macro F1 and error analysis.'),
('Is test data used during training?','No. It remains locked until final evaluation.'),
('Does a high F1 guarantee a perfect model?','No. It omits calibration, subgroup behavior, severity of specific errors and distribution shift.'),
('Does Stage 2 1.0 ROC-AUC prove real performance?','No. It is a synthetic held-out result under an unusually separable generator.'),
('Does feature importance mean causation?','No. It shows model reliance or association, not a biological cause.'),
('Can you call ResNet-50 the final winner?','No, not without completed fair experiments and validation-based selection.'),
('Can the API replace an oncologist?','No. It is a research prototype and must not make autonomous clinical decisions.'),
])

rapid=[
('What is Stage 1 target?','toxicity_risk: Low, Moderate or High.'),('Raw Stage 1 rows?','8,901.'),('Cleaned Stage 1 rows?','8,754.'),('Duplicates removed?','147.'),('Unique Stage 1 patients?','6,000.'),('Best Stage 1 model?','V4 regularized LightGBM.'),('Locked-test macro F1?','0.5288.'),('Locked-test Accuracy?','0.5766.'),('High-risk Recall?','0.6287.'),('Why Recall?','To reduce missed actual High-risk cases.'),('What is a false negative?','An actual target-class case predicted as another class.'),('Critical High-to-Low misses?','50.'),('Hardest class?','Moderate; F1 0.3271.'),('CV method?','5-fold StratifiedGroupKFold by patient_id.'),('Leakage column excluded?','treatment_response.'),('Raw data?','Original uncorrected records.'),('Cleaned data?','Validated, standardized records.'),('Feature data?','Selected/engineered/transformed model input.'),('Engineered feature count?','11.'),('Total Stage 1 features?','41 before expanded encoding.'),('Numeric missing strategy?','Median imputation.'),('Categorical missing strategy?','Unknown or documented defaults.'),('Why remove duplicates?','Prevent repeated influence and optimistic evaluation.'),('Why macro F1?','Equal weight to all three classes.'),('Why not Accuracy alone?','It can hide minority-class failures.'),('What is overfitting?','Strong training performance but weak unseen-patient performance.'),('Example of overfitting?','XGBoost train macro F1 0.7763 vs CV 0.5427.'),('Stage 1 backend?','FastAPI.'),('Stage 1 endpoints?','/health, /predict, /predict/batch.'),('Database?','None implemented.'),('Stage 2 patients?','1,000 synthetic patients.'),('Pathology tiles?','12,000; 12 per patient.'),('Temporal observations?','16,012.'),('Stage 2 split?','700/150/150 patients.'),('Pathology baseline?','Pretrained ResNet-18 with frozen backbone and custom head.'),('Temporal baseline?','Two-layer bidirectional LSTM.'),('Temporal inputs?','Five biomarkers, time features, velocity and five masks.'),('Historical window?','Day 0-90.'),('Forecast target window?','Day 91+.'),('Pathology classes?','Benign, malignant and inflammation.'),('Temporal outputs?','30-day ctDNA and progression probability.'),('Pathology test size?','1,800 tiles.'),('Temporal test size?','150 patients.'),('ctDNA MAE?','0.3698.'),('ctDNA RMSE?','0.5060.'),('ctDNA R2?','0.8236.'),('Why perfect classification?','Synthetic deterministic separation.'),('Fusion weights?','0.35 pathology, 0.40 progression, 0.25 ctDNA.'),('Missing modality states?','FULL_MULTIMODAL, PATHOLOGY_ONLY, TEMPORAL_ONLY, INSUFFICIENT_DATA.'),('Clinical status?','Research-only; not clinically validated.')]
page(story,'PART 14: RAPID-FIRE ROUND (1-25)')
for i,(q,a) in enumerate(rapid[:25],1): story.extend(qa(f'{i}. {q}',a))
page(story,'PART 14: RAPID-FIRE ROUND (26-50)')
for i,(q,a) in enumerate(rapid[25:],26): story.extend(qa(f'{i}. {q}',a))

page(story,'PART 15: SINGLE-PAGE PROJECT DEFENCE SHEET')
facts=[
('PROJECT PROBLEM','Predict oncology toxicity risk; explore synthetic pathology and temporal progression forecasting.'),('DATASET','Stage 1: 8,754 cleaned encounters/6,000 patients. Stage 2: 1,000 synthetic patients, 12,000 tiles, 16,012 observations.'),('RAW DATA','8,901-row oncology_uncleaned.csv with duplicates, missing tokens, unit text, category variants and invalid values.'),('CLEANED DATA','master_patient_dataset.csv; 147 duplicates removed, representations standardized, missing/invalid values handled.'),('FEATURE DATA','Thirty raw predictors + 11 engineered Stage 1 features, encoded through a train-fitted pipeline.'),('TARGET','Stage 1 toxicity_risk. Stage 2 pathology class, progression and 30-day ctDNA.'),('PREPROCESSING','Deduplication, bounds checks, imputation, categorical encoding; image resize/normalization; temporal train-only standardization and masks.'),('EDA','Quality, distributions, correlations, imbalance, chronology, missingness and image-shortcut audits.'),('ML MODELS','LightGBM and XGBoost candidates; V4 LightGBM selected.'),('DL MODELS','Validated baselines: ResNet-18 and 2-layer BiLSTM.'),('TRAINING','Patient-grouped/fixed patient-disjoint splits; train-only fitted preprocessing; validation selection; locked test.'),('METRICS','Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, log loss/Brier; MAE, RMSE, R2.'),('WHY METRICS','Macro F1 and High-risk recall expose minority-class safety failures hidden by Accuracy.'),('BEST MODEL','Stage 1 V4 regularized LightGBM. Stage 2 reports separate validated baseline tasks.'),('RESULT','Stage 1 macro F1 0.5288, High-risk recall 0.6287. Stage 2 ctDNA MAE 0.3698/R2 0.8236; synthetic classification 1.0.'),('ENGINEERING','CSV pipelines and manifests; EDA audits; model training; locked evaluation; FastAPI and Stage 2 Streamlit/report integration.'),('FINAL OUTPUT','Risk class/probabilities; Stage 2 progression probability, ctDNA forecast and modality-aware fusion state.'),('LIMITATION','Research prototype; Stage 2 is synthetic; no clinical deployment or database.')]
t=Table([[P('<b>'+a+'</b>','Small'),P(b,'Small')] for a,b in facts],colWidths=[3.6*cm,13*cm])
t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.22,colors.HexColor('#BECACC')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),('BACKGROUND',(0,0),(0,-1),LIGHT),('ROWBACKGROUNDS',(1,0),(1,-1),[colors.white,colors.HexColor('#F7F9F9')])]))
story += [t]

top25=[
'Raw vs cleaned vs feature data?','Why not train on raw data?','Why focus on High-risk Recall?','Why is Accuracy insufficient?','What is a false negative in Stage 1?','Why macro F1?','Why V4 LightGBM?','Why reject XGBoost?','Why patient-grouped CV?','How did you prevent leakage?','Why exclude treatment_response?','Why median imputation?','Why feature engineering?','What did EDA change?','How do you prove generalization?','What is the locked-test result?','Why is Moderate hardest?','Explain ResNet-18 architecture.','Explain BiLSTM and dual heads.','Why Day 0-90 only?','Why are Stage 2 scores perfect?','How does fixed fusion work?','What if a modality is missing?','How does the API work?','What is the biggest limitation?']
page(story,'TOP 25 QUESTIONS YOU MUST ANSWER')
for i,q in enumerate(top25,1): story.append(P(f'<b>{i}. {q}</b>','A'))
story += [Spacer(1,.25*cm),box('Speaking pattern: 1) define in one sentence, 2) connect to an exact project decision or metric, 3) state the limitation. Example: Recall measures detected actual positives; High-risk recall was 0.6287; it matters because false negatives miss High-risk encounters, but I still balance it with precision and macro F1.')]

doc=SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=1.65*cm,rightMargin=1.65*cm,topMargin=1.35*cm,bottomMargin=1.5*cm)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)
