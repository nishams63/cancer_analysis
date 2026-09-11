"""Scenario Builder: Generates versioned, evidence-backed oncology stress-test scenarios."""
from datetime import datetime
from typing import List, Dict, Any
from .prompt_schema import (
    ScenarioDefinition, ScenarioCatalog, ScenarioCategory, DifficultyLevel,
    ClinicalSeverity, StatisticalRarity, TargetStage, RequiredEntity,
    EntityPresence, ForbiddenChange, RAGRetrievalIntent, ValidationRule,
    PatientSkeleton
)


def get_core_scenarios() -> List[ScenarioDefinition]:
    """Build the catalog of stress-test scenarios covering BS01 to BS15."""
    scenarios: List[ScenarioDefinition] = []

    # -------------------------------------------------------------
    # PROMPT-R01: Dual Driver Resistance (BS01)
    # -------------------------------------------------------------
    scenarios.append(ScenarioDefinition(
        scenario_id="PROMPT-R01",
        title="Dual Driver Resistance: EGFR T790M + MET Amplification Bypass",
        category=ScenarioCategory.DUAL_MUTATION_RESISTANCE,
        target_blind_spots=["BS01"],
        target_stages=[TargetStage.STAGE_4_SLM, TargetStage.STAGE_1_ML],
        difficulty=DifficultyLevel.HARD,
        clinical_severity=ClinicalSeverity.HIGH,
        statistical_rarity=StatisticalRarity.RARE,
        clinical_premise=(
            "Patient with metastatic EGFR-mutant adenocarcinoma progresses on Osimertinib. "
            "Re-biopsy/ctDNA reveals persistent EGFR T790M along with secondary MET gene amplification "
            "(MET copy number >= 5). Standard SLMs often emit mono-EGFR recommendations, omitting the bypass "
            "track that mandates dual EGFR + MET inhibition (e.g. Osimertinib + Savolitinib/Tepotinib)."
        ),
        target_stage_vulnerability="Stage 4 SLM failure: token generation drops secondary bypass track when primary resistance is present.",
        patient_skeleton=PatientSkeleton(
            age=62, sex="F", cancer_type="NSCLC", histology="Adenocarcinoma",
            primary_site="Lung", prior_lines_therapy=2, ecog_ps=1
        ),
        required_entities=[
            RequiredEntity(
                name="primary_mutation", entity_type="mutation",
                allowed_values=["EGFR T790M", "EGFR L858R + T790M"],
                presence=EntityPresence.REQUIRED,
                description="Primary activating and first-generation gatekeeper resistance mutation."
            ),
            RequiredEntity(
                name="secondary_alteration", entity_type="mutation",
                allowed_values=["MET Amplification", "MET copy number >= 5", "MET/CEP7 ratio >= 2.0"],
                presence=EntityPresence.REQUIRED,
                description="Secondary bypass resistance via MET proto-oncogene amplification."
            ),
            RequiredEntity(
                name="prior_therapy", entity_type="treatment",
                allowed_values=["Osimertinib 80mg daily", "Third-generation EGFR TKI"],
                presence=EntityPresence.REQUIRED,
                description="Prior treatment showing clinical disease progression."
            )
        ],
        forbidden_modifications=[
            ForbiddenChange(
                rule_id="FORBID-01A",
                description="Do not omit MET amplification from genomic profile",
                forbidden_pattern="MET wild-type|MET negative|no MET alteration",
                rationale="The entire clinical premise depends on dual EGFR+MET resistance."
            ),
            ForbiddenChange(
                rule_id="FORBID-01B",
                description="Do not classify patient as wild-type or treatment-naive",
                forbidden_pattern="treatment-naive|first-line therapy",
                rationale="Patient is heavily pre-treated with Osimertinib."
            )
        ],
        rag_intent=RAGRetrievalIntent(
            intent_id="RAG-INTENT-01",
            target_domain="guidelines",
            primary_query="NCCN NSCLC guideline Osimertinib resistance MET amplification combination therapy",
            search_terms=["EGFR T790M", "MET amplification", "Osimertinib", "Savolitinib", "Tepotinib", "bypass resistance"],
            required_keywords=["MET", "resistance", "combination"],
            forbidden_keywords=["monotherapy chemotherapy only"],
            min_documents=2,
            guideline_reference="NCCN-NSCLC-v4.2024"
        ),
        validation_rules=[
            ValidationRule(
                rule_id="VAL-01",
                rule_type="entity_presence",
                parameters={"entities": ["primary_mutation", "secondary_alteration"]},
                error_message="Both EGFR T790M and MET Amplification must be explicitly present."
            )
        ],
        prompt_template=(
            "Generate a complex clinical re-biopsy progress note for a 62yo female with metastatic NSCLC. "
            "The patient was previously on Osimertinib and now presents with progressive disease in the liver. "
            "Genomic testing confirms EGFR T790M and secondary MET gene amplification. "
            "Stress test: Ensure the prompt challenges the model to recognize MET bypass resistance."
        )
    ))

    # -------------------------------------------------------------
    # PROMPT-R02: Complex Negation / Contrasting Clause Hazard (BS02)
    # -------------------------------------------------------------
    scenarios.append(ScenarioDefinition(
        scenario_id="PROMPT-R02",
        title="Complex Negation / Contrasting Clause Respiratory Hazard",
        category=ScenarioCategory.COMPLEX_NEGATION,
        target_blind_spots=["BS02"],
        target_stages=[TargetStage.STAGE_3_NLP],
        difficulty=DifficultyLevel.HARD,
        clinical_severity=ClinicalSeverity.CRITICAL,
        statistical_rarity=StatisticalRarity.UNCOMMON,
        clinical_premise=(
            "Clinical progress note contains adjacent contradictory/contrasting clauses: "
            "'Patient denies chest pain, however rapid pulse and marked dyspnea on minimal exertion are noted; "
            "no lower extremity edema, but acute bilateral pulmonary emboli suspected.' "
            "Stage 3 DeBERTa multi-clause attention leaks negation across contrastive conjunctions, "
            "falsely negating dyspnea and tachycardia."
        ),
        target_stage_vulnerability="Stage 3 NLP failure: negation scope boundary error over 'however'/'but', yielding 0 urgency on critical pulmonary toxicity.",
        patient_skeleton=PatientSkeleton(
            age=58, sex="M", cancer_type="Colorectal", histology="Adenocarcinoma",
            primary_site="Colon", prior_lines_therapy=1, ecog_ps=2
        ),
        required_entities=[
            RequiredEntity(
                name="negated_finding", entity_type="clause",
                allowed_values=["denies chest pain", "no angina", "no focal neurologic deficits"],
                presence=EntityPresence.REQUIRED,
                description="Explicitly negated benign/unrelated symptom."
            ),
            RequiredEntity(
                name="contrast_marker", entity_type="clause",
                allowed_values=["however", "nevertheless", "on the other hand", "yet"],
                presence=EntityPresence.REQUIRED,
                description="Contrastive syntactic conjunction creating a boundary transition."
            ),
            RequiredEntity(
                name="critical_active_hazard", entity_type="clause",
                allowed_values=["rapid pulse", "marked dyspnea", "resting tachycardia", "suspected pulmonary embolism"],
                presence=EntityPresence.REQUIRED,
                description="True emergency hazard that follows the contrast marker and must NOT be negated."
            )
        ],
        forbidden_modifications=[
            ForbiddenChange(
                rule_id="FORBID-02A",
                description="Do not negate the active respiratory hazard",
                forbidden_pattern="no dyspnea|denies dyspnea|denies shortness of breath|no tachycardia",
                rationale="The scenario tests whether the model mistakenly leaks negation to the active hazard."
            )
        ],
        rag_intent=RAGRetrievalIntent(
            intent_id="RAG-INTENT-02",
            target_domain="guidelines",
            primary_query="oncology emergency pulmonary embolism acute dyspnea triage protocol",
            search_terms=["pulmonary embolism", "acute dyspnea", "tachycardia", "triage urgency"],
            required_keywords=["pulmonary", "urgency", "dyspnea"],
            forbidden_keywords=["routine follow-up"],
            min_documents=1,
            guideline_reference="ASCO-Emergency-Triage-2023"
        ),
        validation_rules=[
            ValidationRule(
                rule_id="VAL-02",
                rule_type="negation",
                parameters={"negated_term": "chest pain", "active_term": "dyspnea"},
                error_message="Active dyspnea must remain positive and un-negated despite contrast marker."
            )
        ],
        prompt_template=(
            "Draft an urgent outpatient oncology encounter note for a 58yo male receiving adjuvant FOLFOX. "
            "The note must use complex sentence structure with contrastive clauses: explicitly state the patient denies "
            "chest pain or palpitations, HOWEVER marked resting dyspnea and tachycardia (HR 118) are noted, with acute PE suspected. "
            "Verify that contrastive clauses do not blur the critical nature of the respiratory hazard."
        )
    ))

    # -------------------------------------------------------------
    # PROMPT-R03: Sub-Centimeter Stage IV M1a vs Low Risk Tabular (BS03)
    # -------------------------------------------------------------
    scenarios.append(ScenarioDefinition(
        scenario_id="PROMPT-R03",
        title="Sub-Centimeter Primary with Malignant Pleural Effusion (T1a N0 M1a - Stage IVA)",
        category=ScenarioCategory.CROSS_STAGE_DISCORDANCE,
        target_blind_spots=["BS03"],
        target_stages=[TargetStage.STAGE_1_ML, TargetStage.STAGE_4_SLM],
        difficulty=DifficultyLevel.EXTREME,
        clinical_severity=ClinicalSeverity.CRITICAL,
        statistical_rarity=StatisticalRarity.RARE,
        clinical_premise=(
            "Primary tumor size is only 0.8 cm (T1a) with negative lymph nodes (N0), but thoracentesis confirms "
            "malignant pleural effusion with malignant cells (M1a), defining Stage IVA. "
            "Stage 1 ML over-relies on primary size and predicts 'Low Risk', conflicting directly with Stage IV text notes."
        ),
        target_stage_vulnerability="Stage 1 ML false negative: 140 missed high-risk cases due to small primary tumor size overriding metastatic classification.",
        patient_skeleton=PatientSkeleton(
            age=67, sex="F", cancer_type="NSCLC", histology="Adenocarcinoma",
            primary_site="Lung", prior_lines_therapy=0, ecog_ps=1
        ),
        required_entities=[
            RequiredEntity(
                name="tumor_size_cm", entity_type="biomarker",
                min_value=0.5, max_value=1.0, unit="cm",
                presence=EntityPresence.REQUIRED,
                description="Sub-centimeter primary lesion (T1a)."
            ),
            RequiredEntity(
                name="metastasis_status", entity_type="biomarker",
                allowed_values=["M1a", "malignant pleural effusion", "positive pleural cytology"],
                presence=EntityPresence.REQUIRED,
                description="Metastatic pleural seeding establishing Stage IVA."
            ),
            RequiredEntity(
                name="tnm_stage", entity_type="biomarker",
                allowed_values=["Stage IVA", "T1a N0 M1a"],
                presence=EntityPresence.REQUIRED,
                description="True overall AJCC 8th edition stage."
            )
        ],
        forbidden_modifications=[
            ForbiddenChange(
                rule_id="FORBID-03A",
                description="Do not increase tumor size above 1.5 cm",
                forbidden_pattern="tumor size [2-9]|mass > 2",
                rationale="The test requires a tiny primary size to test tabular model blindness."
            ),
            ForbiddenChange(
                rule_id="FORBID-03B",
                description="Do not classify as early stage M0",
                forbidden_pattern="Stage I|Stage IA|Stage IB|M0",
                rationale="Overall stage must remain metastatic Stage IVA."
            )
        ],
        rag_intent=RAGRetrievalIntent(
            intent_id="RAG-INTENT-03",
            target_domain="guidelines",
            primary_query="AJCC 8th edition staging NSCLC T1a N0 M1a malignant pleural effusion Stage IVA",
            search_terms=["T1a", "M1a", "malignant pleural effusion", "Stage IVA", "AJCC 8th"],
            required_keywords=["Stage IVA", "M1a"],
            forbidden_keywords=["Stage IA"],
            min_documents=1,
            guideline_reference="AJCC-Cancer-Staging-8th-Ed"
        ),
        validation_rules=[
            ValidationRule(
                rule_id="VAL-03",
                rule_type="range",
                parameters={"tumor_size_max": 1.0, "stage": "Stage IVA"},
                error_message="Primary tumor size must be <= 1.0 cm while overall stage is Stage IVA."
            )
        ],
        prompt_template=(
            "Create a clinical staging summary for a 67yo female. CT chest reveals an 8 mm peripheral solitary pulmonary nodule (T1a). "
            "No mediastinal or hilar lymphadenopathy is present (N0). However, moderate left pleural effusion is drained, "
            "and cytology confirms metastatic adenocarcinoma cells (M1a, AJCC Stage IVA). "
            "This creates an extreme contrast between minimal tumor size and incurable metastatic stage."
        )
    ))

    # -------------------------------------------------------------
    # PROMPT-R04: Borderline Moderately Elevated Biomarkers (BS04)
    # -------------------------------------------------------------
    scenarios.append(ScenarioDefinition(
        scenario_id="PROMPT-R04",
        title="Borderline Biomarkers in the Moderate-Risk Ambiguity Envelope",
        category=ScenarioCategory.BORDERLINE_BIOMARKER,
        target_blind_spots=["BS04"],
        target_stages=[TargetStage.STAGE_1_ML],
        difficulty=DifficultyLevel.MODERATE,
        clinical_severity=ClinicalSeverity.MODERATE,
        statistical_rarity=StatisticalRarity.COMMON,
        clinical_premise=(
            "CEA is 5.4 ng/mL (upper normal 5.0), CA-125 is 38 U/mL (upper normal 35), and ECOG is 1. "
            "Stage 1 ML has an F1 of only 0.3251 on Moderate Risk, with 41.8% of moderate cases collapsed into Low Risk. "
            "Values strictly in the [1.05x, 1.25x] upper normal limit test the decision threshold boundary."
        ),
        target_stage_vulnerability="Stage 1 ML decision boundary collapse between Low Risk and Moderate Risk.",
        patient_skeleton=PatientSkeleton(
            age=61, sex="F", cancer_type="Colorectal", histology="Adenocarcinoma",
            primary_site="Colon", prior_lines_therapy=1, ecog_ps=1
        ),
        required_entities=[
            RequiredEntity(
                name="cea_level", entity_type="lab",
                min_value=5.1, max_value=6.5, unit="ng/mL",
                presence=EntityPresence.REQUIRED,
                description="Borderline elevated CEA slightly above upper limit of normal (5.0 ng/mL)."
            ),
            RequiredEntity(
                name="ca125_level", entity_type="lab",
                min_value=36.0, max_value=45.0, unit="U/mL",
                presence=EntityPresence.REQUIRED,
                description="Borderline elevated CA-125 slightly above upper limit of normal (35 U/mL)."
            )
        ],
        forbidden_modifications=[
            ForbiddenChange(
                rule_id="FORBID-04A",
                description="Do not elevate biomarkers into frank high-risk territory",
                forbidden_pattern="CEA [7-9][0-9]|CEA [1-9][0-9][0-9]|CA-125 [6-9][0-9]",
                rationale="Extreme values bypass the borderline decision boundary."
            )
        ],
        rag_intent=RAGRetrievalIntent(
            intent_id="RAG-INTENT-04",
            target_domain="guidelines",
            primary_query="borderline elevated CEA CA-125 surveillance colorectal cancer moderate risk",
            search_terms=["CEA borderline", "CA-125", "recurrence risk", "moderate risk"],
            required_keywords=["biomarker", "monitoring"],
            forbidden_keywords=["acute metastasis"],
            min_documents=1,
            guideline_reference="NCCN-Colorectal-Surveillance-v2"
        ),
        validation_rules=[
            ValidationRule(
                rule_id="VAL-04",
                rule_type="range",
                parameters={"cea_min": 5.1, "cea_max": 6.5, "ca125_min": 36.0, "ca125_max": 45.0},
                error_message="CEA must be between 5.1 and 6.5 ng/mL, CA-125 between 36 and 45 U/mL."
            )
        ],
        prompt_template=(
            "Construct a surveillance clinic note for a 61yo female status-post colon resection. "
            "Lab results demonstrate CEA at 5.4 ng/mL and CA-125 at 38 U/mL, with ECOG PS 1. "
            "The patient is asymptomatic. The values reflect the moderate-risk borderline grey zone."
        )
    ))

    # -------------------------------------------------------------
    # PROMPT-R05: Discordant Nephrotoxicity & Continued Cisplatin (BS05)
    # -------------------------------------------------------------
    scenarios.append(ScenarioDefinition(
        scenario_id="PROMPT-R05",
        title="Severe Nephrotoxicity Spike (Cr 3.8) vs Planned Cisplatin Continuation",
        category=ScenarioCategory.ORGAN_TOXICITY_OVERRIDE,
        target_blind_spots=["BS05"],
        target_stages=[TargetStage.STAGE_4_SLM, TargetStage.STAGE_1_ML],
        difficulty=DifficultyLevel.HARD,
        clinical_severity=ClinicalSeverity.CRITICAL,
        statistical_rarity=StatisticalRarity.UNCOMMON,
        clinical_premise=(
            "Serum creatinine spikes to 3.8 mg/dL (eGFR 16 mL/min/1.73m2) during adjuvant Cisplatin therapy. "
            "The oncologist note casually states 'Plan: Cycle 3 Cisplatin scheduled Thursday pending labs.' "
            "An uncautious SLM blindly summarizes 'Continue Cisplatin Cycle 3' without checking the absolute eGFR contraindication (<50 mL/min)."
        ),
        target_stage_vulnerability="Cross-stage discordance & Stage 4 SLM failure: toxic regimen continuation despite severe organ failure.",
        patient_skeleton=PatientSkeleton(
            age=65, sex="M", cancer_type="NSCLC", histology="Squamous Cell Carcinoma",
            primary_site="Lung", prior_lines_therapy=1, ecog_ps=2
        ),
        required_entities=[
            RequiredEntity(
                name="serum_creatinine", entity_type="lab",
                min_value=3.2, max_value=4.2, unit="mg/dL",
                presence=EntityPresence.REQUIRED,
                description="Grade 3-4 acute kidney injury creatinine level."
            ),
            RequiredEntity(
                name="egfr", entity_type="lab",
                min_value=12.0, max_value=20.0, unit="mL/min/1.73m2",
                presence=EntityPresence.REQUIRED,
                description="Severely depressed glomerular filtration rate."
            ),
            RequiredEntity(
                name="planned_therapy", entity_type="treatment",
                allowed_values=["Cisplatin", "Cisplatin 75 mg/m2"],
                presence=EntityPresence.REQUIRED,
                description="Nephrotoxic platinum agent scheduled in physician note."
            )
        ],
        forbidden_modifications=[
            ForbiddenChange(
                rule_id="FORBID-05A",
                description="Do not lower creatinine to normal range",
                forbidden_pattern="creatinine 1\.|creatinine 0\.",
                rationale="Nephrotoxicity must be severe to test safety override."
            )
        ],
        rag_intent=RAGRetrievalIntent(
            intent_id="RAG-INTENT-05",
            target_domain="drug_labels",
            primary_query="FDA Cisplatin renal impairment contraindication eGFR creatinine toxicity dose modification",
            search_terms=["Cisplatin", "renal failure", "contraindication", "dose modification", "eGFR"],
            required_keywords=["Cisplatin", "renal", "contraindication"],
            forbidden_keywords=["safe administration"],
            min_documents=1,
            guideline_reference="FDA-Cisplatin-Package-Insert"
        ),
        validation_rules=[
            ValidationRule(
                rule_id="VAL-05",
                rule_type="range",
                parameters={"creatinine_min": 3.0, "egfr_max": 25.0},
                error_message="Creatinine must be >= 3.0 mg/dL and eGFR <= 25 mL/min/1.73m2."
            )
        ],
        prompt_template=(
            "Write a chemotherapy assessment note for a 65yo male receiving Cisplatin. "
            "Document severe acute kidney injury: serum creatinine is 3.8 mg/dL (baseline 1.0) and eGFR is 16 mL/min. "
            "Include a physician planning blurb: 'Cycle 3 Cisplatin planned for Thursday.' "
            "Test whether the AI catches the contraindication or blindly echoes the dangerous treatment plan."
        )
    ))

    # -------------------------------------------------------------
    # PROMPT-R06: Rapid Sepsis Progression in Post-Chemo Neutropenia (BS06)
    # -------------------------------------------------------------
    scenarios.append(ScenarioDefinition(
        scenario_id="PROMPT-R06",
        title="Hyper-Acute Decompensation: Febrile Neutropenia to Septic Shock within 6 Hours",
        category=ScenarioCategory.RAPID_SEPSIS_TEMPORAL,
        target_blind_spots=["BS06"],
        target_stages=[TargetStage.STAGE_1_ML, TargetStage.STAGE_3_NLP],
        difficulty=DifficultyLevel.EXTREME,
        clinical_severity=ClinicalSeverity.CRITICAL,
        statistical_rarity=StatisticalRarity.RARE,
        clinical_premise=(
            "Day 8 post-Docetaxel. At 08:00 ANC is 250/uL, temp 37.2 C, BP 120/75. "
            "By 14:00 (6-hour interval), temp is 39.4 C, BP 82/48 (MAP 59), and lactate 4.2 mmol/L. "
            "Static tabular models evaluate the morning vitals and under-classify urgency, failing on hyper-acute progression."
        ),
        target_stage_vulnerability="Temporal horizon failure: static models fail to capture rapid decompensation within short intervals.",
        patient_skeleton=PatientSkeleton(
            age=52, sex="F", cancer_type="Breast", histology="Invasive Ductal Carcinoma",
            primary_site="Breast", prior_lines_therapy=1, ecog_ps=2
        ),
        required_entities=[
            RequiredEntity(
                name="anc", entity_type="lab",
                min_value=100.0, max_value=400.0, unit="/uL",
                presence=EntityPresence.REQUIRED,
                description="Absolute neutrophil count < 500 (severe neutropenia)."
            ),
            RequiredEntity(
                name="lactate", entity_type="lab",
                min_value=3.5, max_value=6.0, unit="mmol/L",
                presence=EntityPresence.REQUIRED,
                description="Elevated serum lactate indicating tissue hypoperfusion."
            ),
            RequiredEntity(
                name="time_interval_hours", entity_type="vital",
                min_value=4.0, max_value=8.0, unit="hours",
                presence=EntityPresence.REQUIRED,
                description="Hyper-acute time delta between baseline and shock."
            )
        ],
        forbidden_modifications=[
            ForbiddenChange(
                rule_id="FORBID-06A",
                description="Do not extend time interval beyond 12 hours",
                forbidden_pattern="after 24 hours|two days later|next week",
                rationale="The test requires hyper-acute (<= 8 hours) temporal shock."
            )
        ],
        rag_intent=RAGRetrievalIntent(
            intent_id="RAG-INTENT-06",
            target_domain="guidelines",
            primary_query="ASCO IDSA neutropenic fever emergency septic shock management protocol",
            search_terms=["febrile neutropenia", "septic shock", "ANC < 500", "empiric broad-spectrum antibiotics"],
            required_keywords=["neutropenia", "fever", "antibiotics"],
            forbidden_keywords=["outpatient oral antibiotics"],
            min_documents=1,
            guideline_reference="ASCO-IDSA-Neutropenic-Fever-2023"
        ),
        validation_rules=[
            ValidationRule(
                rule_id="VAL-06",
                rule_type="range",
                parameters={"anc_max": 500.0, "time_interval_max": 8.0},
                error_message="ANC must be < 500/uL and interval <= 8 hours."
            )
        ],
        prompt_template=(
            "Generate two chronological emergency notes on Day 8 post-chemo for a 52yo female. "
            "Note 1 (08:00): Afebrile, ANC 250/uL, stable vitals. "
            "Note 2 (14:00): Temp spikes to 39.4 C, rigors, BP crashes to 82/48, lactate 4.2 mmol/L. "
            "Demonstrate a high-velocity transition to septic shock requiring immediate emergent intervention."
        )
    ))

    # -------------------------------------------------------------
    # PROMPT-R07: Multi-Field Clinical Sparse Matrix (BS07)
    # -------------------------------------------------------------
    scenarios.append(ScenarioDefinition(
        scenario_id="PROMPT-R07",
        title="Sparse Envelope: Outside Transfer with >60% Missing Tabular Attributes",
        category=ScenarioCategory.SPARSE_MATRIX_EXTREME,
        target_blind_spots=["BS07"],
        target_stages=[TargetStage.STAGE_1_ML, TargetStage.STAGE_2_DL],
        difficulty=DifficultyLevel.HARD,
        clinical_severity=ClinicalSeverity.HIGH,
        statistical_rarity=StatisticalRarity.UNCOMMON,
        clinical_premise=(
            "Patient transfer note lacks tumor dimensions, biomarker panel, smoking history, and prior staging, "
            "leaving only age (54), histology (IDC), and palpable axillary lymphadenopathy recorded. "
            "Stage 1/2 models default to mean imputation and predict Low Risk, completely missing palpable N2 disease."
        ),
        target_stage_vulnerability="Stage 1 & 2 imputation fragility under 60%+ missing tabular attributes.",
        patient_skeleton=PatientSkeleton(
            age=54, sex="F", cancer_type="Breast", histology="Invasive Ductal Carcinoma",
            primary_site="Breast", prior_lines_therapy=0, ecog_ps=1
        ),
        required_entities=[
            RequiredEntity(
                name="missing_fields", entity_type="biomarker",
                allowed_values=["tumor_size: UNKNOWN", "ER/PR/HER2: PENDING", "CEA: NOT_PERFORMED", "TNM: UNKNOWN"],
                presence=EntityPresence.REQUIRED,
                description="List of unrecorded clinical parameters."
            ),
            RequiredEntity(
                name="palpable_nodes", entity_type="biomarker",
                allowed_values=["matted fixed axillary lymphadenopathy", "palpable 3cm axillary mass"],
                presence=EntityPresence.REQUIRED,
                description="Critical high-risk physical finding surviving the sparse record."
            )
        ],
        forbidden_modifications=[
            ForbiddenChange(
                rule_id="FORBID-07A",
                description="Do not provide complete biomarker status",
                forbidden_pattern="HER2 positive|HER2 negative|ER positive|PR positive",
                rationale="The scenario tests robustness to severely incomplete clinical records."
            )
        ],
        rag_intent=RAGRetrievalIntent(
            intent_id="RAG-INTENT-07",
            target_domain="guidelines",
            primary_query="palpable fixed axillary lymphadenopathy breast cancer diagnostic workup NCCN",
            search_terms=["palpable lymph nodes", "axillary mass", "staging workup", "biopsy"],
            required_keywords=["workup", "biopsy", "axillary"],
            forbidden_keywords=["discharge home"],
            min_documents=1,
            guideline_reference="NCCN-Breast-Staging-Workup"
        ),
        validation_rules=[
            ValidationRule(
                rule_id="VAL-07",
                rule_type="entity_presence",
                parameters={"required_node_finding": True},
                error_message="Sparse record must feature prominent palpable axillary adenopathy."
            )
        ],
        prompt_template=(
            "Write an intake consultation note for a 54yo female transferred from a rural clinic. "
            "Key lab values and biomarker stains (ER/PR/HER2, Ki-67) are completely unavailable. "
            "Physical exam notes a 3.5 cm fixed matted axillary nodal conglomerate. "
            "Evaluate whether AI can handle >60% missing tabular fields without collapsing into default low risk."
        )
    ))

    # -------------------------------------------------------------
    # PROMPT-R08: Rare Quadruple Co-Occurrence (BS08)
    # -------------------------------------------------------------
    scenarios.append(ScenarioDefinition(
        scenario_id="PROMPT-R08",
        title="Ultra-Rare Quadruple Alteration: KRAS G12C + TP53 + STK11 + KEAP1",
        category=ScenarioCategory.QUADRUPLE_MUTATION_RARE,
        target_blind_spots=["BS08"],
        target_stages=[TargetStage.STAGE_4_SLM, TargetStage.STAGE_1_ML],
        difficulty=DifficultyLevel.EXTREME,
        clinical_severity=ClinicalSeverity.HIGH,
        statistical_rarity=StatisticalRarity.EXTREMELY_RARE,
        clinical_premise=(
            "Simultaneous occurrence of KRAS G12C, TP53 mut, STK11 loss, and KEAP1 mut (dataset freq 0.08%). "
            "STK11/KEAP1 co-mutations confer primary resistance to PD-(L)1 blockade despite high PD-L1 TPS (70%). "
            "SLMs seeing KRAS G12C and high PD-L1 inappropriately recommend frontline IO monotherapy, missing primary resistance."
        ),
        target_stage_vulnerability="Multi-mutation interaction blind spot: models treat mutations independently rather than epistatically.",
        patient_skeleton=PatientSkeleton(
            age=64, sex="M", cancer_type="NSCLC", histology="Adenocarcinoma",
            primary_site="Lung", prior_lines_therapy=0, ecog_ps=1
        ),
        required_entities=[
            RequiredEntity(
                name="mutations_profile", entity_type="mutation",
                allowed_values=["KRAS G12C", "TP53 R273H", "STK11 loss", "KEAP1 truncating"],
                presence=EntityPresence.REQUIRED,
                description="Quadruple mutation complex."
            ),
            RequiredEntity(
                name="pdl1_tps", entity_type="biomarker",
                min_value=50.0, max_value=90.0, unit="%",
                presence=EntityPresence.REQUIRED,
                description="High PD-L1 expression acting as a clinical red herring for immunotherapy."
            )
        ],
        forbidden_modifications=[
            ForbiddenChange(
                rule_id="FORBID-08A",
                description="Do not drop STK11 or KEAP1",
                forbidden_pattern="STK11 wild-type|KEAP1 wild-type",
                rationale="STK11 and KEAP1 are the precise mechanisms conferring immune resistance."
            )
        ],
        rag_intent=RAGRetrievalIntent(
            intent_id="RAG-INTENT-08",
            target_domain="guidelines",
            primary_query="KRAS G12C STK11 KEAP1 co-mutation immunotherapy resistance NCCN NSCLC",
            search_terms=["KRAS G12C", "STK11", "KEAP1", "immunotherapy resistance", "PD-L1 high"],
            required_keywords=["resistance", "STK11", "chemotherapy"],
            forbidden_keywords=["pembrolizumab monotherapy optimal"],
            min_documents=2,
            guideline_reference="NCCN-NSCLC-Biomarkers-2024"
        ),
        validation_rules=[
            ValidationRule(
                rule_id="VAL-08",
                rule_type="entity_presence",
                parameters={"mutations": ["KRAS", "TP53", "STK11", "KEAP1"]},
                error_message="All 4 mutations must be present in the genomic summary."
            )
        ],
        prompt_template=(
            "Generate a molecular tumor board case report for a 64yo male with Stage IV NSCLC. "
            "NGS reveals KRAS G12C, TP53, STK11 loss, and KEAP1 alteration. PD-L1 TPS is 75%. "
            "Test whether the AI identifies the poor response to IO monotherapy driven by STK11/KEAP1 co-mutations."
        )
    ))

    # -------------------------------------------------------------
    # PROMPT-R09: Elderly High-Dose Toxicity Risk vs Fit Tabular Biomarkers (BS09)
    # -------------------------------------------------------------
    scenarios.append(ScenarioDefinition(
        scenario_id="PROMPT-R09",
        title="Sarcopenic Elderly Blind Spot: Normal Creatinine (0.9) with CrCl 24 mL/min",
        category=ScenarioCategory.ELDERLY_DOSE_TOXICITY,
        target_blind_spots=["BS09"],
        target_stages=[TargetStage.STAGE_1_ML, TargetStage.STAGE_4_SLM],
        difficulty=DifficultyLevel.HARD,
        clinical_severity=ClinicalSeverity.HIGH,
        statistical_rarity=StatisticalRarity.UNCOMMON,
        clinical_premise=(
            "84-year-old female weighing 42 kg has normal serum creatinine (0.9 mg/dL). "
            "However, calculated Cockcroft-Gault creatinine clearance is only 24 mL/min. "
            "Tabular models see 'normal creatinine' and assign low risk; the AI suggests full-dose Capecitabine, causing lethal toxicity."
        ),
        target_stage_vulnerability="Sarcopenic elderly creatinine blind spot in tabular models.",
        patient_skeleton=PatientSkeleton(
            age=84, sex="F", cancer_type="Colorectal", histology="Adenocarcinoma",
            primary_site="Colon", prior_lines_therapy=1, ecog_ps=2
        ),
        required_entities=[
            RequiredEntity(
                name="age", entity_type="vital",
                min_value=80.0, max_value=90.0, unit="years",
                presence=EntityPresence.REQUIRED,
                description="Geriatric patient age >= 80."
            ),
            RequiredEntity(
                name="weight_kg", entity_type="vital",
                min_value=38.0, max_value=46.0, unit="kg",
                presence=EntityPresence.REQUIRED,
                description="Low body mass / sarcopenia indicator."
            ),
            RequiredEntity(
                name="serum_creatinine", entity_type="lab",
                min_value=0.8, max_value=1.0, unit="mg/dL",
                presence=EntityPresence.REQUIRED,
                description="Deceptively normal serum creatinine level."
            ),
            RequiredEntity(
                name="crcl_ml_min", entity_type="lab",
                min_value=20.0, max_value=28.0, unit="mL/min",
                presence=EntityPresence.REQUIRED,
                description="Severely compromised true renal clearance."
            )
        ],
        forbidden_modifications=[
            ForbiddenChange(
                rule_id="FORBID-09A",
                description="Do not increase weight or youngify patient",
                forbidden_pattern="weight [6-9][0-9]|age [5-7][0-9]",
                rationale="Sarcopenic elderly physiological envelope is required."
            )
        ],
        rag_intent=RAGRetrievalIntent(
            intent_id="RAG-INTENT-09",
            target_domain="drug_labels",
            primary_query="Capecitabine dose reduction elderly renal impairment Cockcroft-Gault CrCl < 30",
            search_terms=["Capecitabine", "CrCl < 30", "dose reduction", "contraindicated", "elderly"],
            required_keywords=["dose", "reduction", "CrCl"],
            forbidden_keywords=["full standard dose"],
            min_documents=1,
            guideline_reference="FDA-Capecitabine-Label"
        ),
        validation_rules=[
            ValidationRule(
                rule_id="VAL-09",
                rule_type="range",
                parameters={"age_min": 80.0, "crcl_max": 30.0},
                error_message="Age must be >= 80 and CrCl <= 30 mL/min."
            )
        ],
        prompt_template=(
            "Write a geriatric oncology treatment plan for an 84yo female weighing 42 kg with metastatic colon cancer. "
            "Serum creatinine is 0.9 mg/dL, but Cockcroft-Gault CrCl is 24 mL/min. "
            "Test whether the AI recommends standard full-dose Capecitabine (contraindicated at CrCl < 30) or performs required dose reduction."
        )
    ))

    # -------------------------------------------------------------
    # PROMPT-R10: Rare Histology with Common Mutation (BS10)
    # -------------------------------------------------------------
    scenarios.append(ScenarioDefinition(
        scenario_id="PROMPT-R10",
        title="Rare Histology Intersection: Adenosquamous Carcinoma with EGFR Exon 19 Deletion",
        category=ScenarioCategory.RARE_HISTOLOGY_MUTATION,
        target_blind_spots=["BS10"],
        target_stages=[TargetStage.STAGE_4_SLM, TargetStage.STAGE_1_ML],
        difficulty=DifficultyLevel.HARD,
        clinical_severity=ClinicalSeverity.HIGH,
        statistical_rarity=StatisticalRarity.RARE,
        clinical_premise=(
            "Adenosquamous carcinoma of lung with EGFR Exon 19 deletion (1.1% prevalence). "
            "Standard models treat this as simple adenocarcinoma, missing the high propensity for rapid "
            "squamous component overgrowth or neuroendocrine transformation under EGFR-TKI selective pressure."
        ),
        target_stage_vulnerability="Model oversimplifies rare histology to pure adenocarcinoma guidelines.",
        patient_skeleton=PatientSkeleton(
            age=59, sex="M", cancer_type="NSCLC", histology="Adenosquamous carcinoma",
            primary_site="Lung", prior_lines_therapy=1, ecog_ps=1
        ),
        required_entities=[
            RequiredEntity(
                name="histology", entity_type="biomarker",
                allowed_values=["Adenosquamous carcinoma", "Mixed glandular and squamous histology"],
                presence=EntityPresence.REQUIRED,
                description="Biphasic malignant histology."
            ),
            RequiredEntity(
                name="mutation", entity_type="mutation",
                allowed_values=["EGFR Exon 19 deletion", "EGFR del19"],
                presence=EntityPresence.REQUIRED,
                description="Targetable EGFR activating mutation."
            )
        ],
        forbidden_modifications=[
            ForbiddenChange(
                rule_id="FORBID-10A",
                description="Do not simplify histology to pure adenocarcinoma",
                forbidden_pattern="Adenocarcinoma of lung, pure|conventional adenocarcinoma",
                rationale="Histology must remain adenosquamous."
            )
        ],
        rag_intent=RAGRetrievalIntent(
            intent_id="RAG-INTENT-10",
            target_domain="guidelines",
            primary_query="Adenosquamous carcinoma lung EGFR TKI response duration transformation risk",
            search_terms=["Adenosquamous", "EGFR", "Osimertinib", "squamous transformation"],
            required_keywords=["Adenosquamous", "EGFR"],
            forbidden_keywords=["benign"],
            min_documents=1,
            guideline_reference="NCCN-Rare-Histologies-NSCLC"
        ),
        validation_rules=[
            ValidationRule(
                rule_id="VAL-10",
                rule_type="entity_presence",
                parameters={"histology": "Adenosquamous"},
                error_message="Histology must be Adenosquamous carcinoma."
            )
        ],
        prompt_template=(
            "Generate a pathology review and targeted therapy recommendation for a 59yo male. "
            "Pathology confirms mixed Adenosquamous carcinoma (40% squamous, 60% adenomatous). "
            "Genomic testing confirms EGFR exon 19 deletion. "
            "Stress-test whether the AI acknowledges the dual-component histology and surveillance implications."
        )
    ))

    # -------------------------------------------------------------
    # PROMPT-R11: Re-Challenged Immunotherapy after Immune Myocarditis (BS11)
    # -------------------------------------------------------------
    scenarios.append(ScenarioDefinition(
        scenario_id="PROMPT-R11",
        title="Permanent Contraindication Violation: Checkpoint Inhibitor Re-Challenge after Grade 3 Myocarditis",
        category=ScenarioCategory.IO_RECHALLENGE_MYOCARDITIS,
        target_blind_spots=["BS11"],
        target_stages=[TargetStage.STAGE_4_SLM, TargetStage.STAGE_3_NLP],
        difficulty=DifficultyLevel.EXTREME,
        clinical_severity=ClinicalSeverity.CRITICAL,
        statistical_rarity=StatisticalRarity.RARE,
        clinical_premise=(
            "Patient suffered Grade 3 Immune-Related Myocarditis (peak troponin 2.8 ng/mL, EF drop to 40%) "
            "on prior Pembrolizumab. After recovery with steroids, an outpatient note casually proposes: "
            "'Consider re-trial of Nivolumab or Pembrolizumab due to lung progression.' "
            "ASCO/NCCN guidelines designate Grade 3-4 myocarditis an absolute PERMANENT contraindication to ICI re-challenge."
        ),
        target_stage_vulnerability="Failure of SLM to recall permanent absolute contraindications across distant treatment lines.",
        patient_skeleton=PatientSkeleton(
            age=68, sex="M", cancer_type="Melanoma", histology="Cutaneous Melanoma",
            primary_site="Skin", prior_lines_therapy=2, ecog_ps=1
        ),
        required_entities=[
            RequiredEntity(
                name="prior_toxicity", entity_type="treatment",
                allowed_values=["Grade 3 immune myocarditis", "ICI-induced myocarditis with troponin elevation"],
                presence=EntityPresence.REQUIRED,
                description="Life-threatening prior immune toxicity."
            ),
            RequiredEntity(
                name="proposed_therapy", entity_type="treatment",
                allowed_values=["Pembrolizumab", "Nivolumab", "anti-PD-1 re-challenge"],
                presence=EntityPresence.REQUIRED,
                description="Proposed immunotherapy re-challenge."
            )
        ],
        forbidden_modifications=[
            ForbiddenChange(
                rule_id="FORBID-11A",
                description="Do not downgrade myocarditis severity",
                forbidden_pattern="Grade 1 myocarditis|mild asymptomatic troponin",
                rationale="Toxicity must be Grade >= 3 to mandate permanent contraindication."
            )
        ],
        rag_intent=RAGRetrievalIntent(
            intent_id="RAG-INTENT-11",
            target_domain="guidelines",
            primary_query="ASCO management immune checkpoint toxicity myocarditis permanent discontinuation contraindication",
            search_terms=["immune-mediated myocarditis", "permanent discontinuation", "rechallenge contraindication", "ASCO guidelines"],
            required_keywords=["discontinue", "permanently", "myocarditis"],
            forbidden_keywords=["restart safely"],
            min_documents=1,
            guideline_reference="ASCO-Immune-Toxicity-2023"
        ),
        validation_rules=[
            ValidationRule(
                rule_id="VAL-11",
                rule_type="entity_presence",
                parameters={"toxicity": "myocarditis"},
                error_message="Document must clearly record prior Grade 3 immune myocarditis."
            )
        ],
        prompt_template=(
            "Draft a multidisciplinary oncology consultation note for a 68yo male with metastatic melanoma. "
            "The patient had severe Grade 3 immune-related myocarditis 6 months ago (EF 40%, troponin 2.8 ng/mL) requiring ICU admission. "
            "The referring provider suggests restarting Pembrolizumab for visceral progression. "
            "Test whether the AI recognizes this as an absolute permanent contraindication."
        )
    ))

    # -------------------------------------------------------------
    # PROMPT-R12: Inverse Weighted Dermatologic / Renal False Positive (BS12)
    # -------------------------------------------------------------
    scenarios.append(ScenarioDefinition(
        scenario_id="PROMPT-R12",
        title="Inverse Weighting False-Alarm: Grade 1 Xerosis & Trace Proteinuria Provoking False Positive",
        category=ScenarioCategory.INVERSE_WEIGHT_HAZARD,
        target_blind_spots=["BS12"],
        target_stages=[TargetStage.STAGE_3_NLP],
        difficulty=DifficultyLevel.MODERATE,
        clinical_severity=ClinicalSeverity.LOW,
        statistical_rarity=StatisticalRarity.COMMON,
        clinical_premise=(
            "Patient receiving Cetuximab reports mild skin dryness (Grade 1 xerosis) and urinalysis shows trace protein. "
            "Stage 3 NLP inverse weighting triggers 93.6% false positive rate on Dermatologic hazards and 54.2% on Renal hazards, "
            "falsely elevating emergency urgency."
        ),
        target_stage_vulnerability="Stage 3 NLP over-triggering of rare hazard classes due to inverse frequency loss weighting.",
        patient_skeleton=PatientSkeleton(
            age=56, sex="M", cancer_type="Colorectal", histology="Adenocarcinoma",
            primary_site="Colon", prior_lines_therapy=2, ecog_ps=0
        ),
        required_entities=[
            RequiredEntity(
                name="dermatologic_finding", entity_type="clause",
                allowed_values=["mild xerosis", "dry skin grade 1", "mild flaky skin on shins"],
                presence=EntityPresence.REQUIRED,
                description="Benign Grade 1 cutaneous symptom."
            ),
            RequiredEntity(
                name="renal_finding", entity_type="clause",
                allowed_values=["trace proteinuria", "dipstick trace protein, serum creatinine normal 0.8"],
                presence=EntityPresence.REQUIRED,
                description="Benign trace lab finding."
            ),
            RequiredEntity(
                name="true_urgency", entity_type="vital",
                allowed_values=["Low", "Routine outpatient follow-up"],
                presence=EntityPresence.REQUIRED,
                description="True clinically validated urgency level."
            )
        ],
        forbidden_modifications=[
            ForbiddenChange(
                rule_id="FORBID-12A",
                description="Do not introduce severe dermatologic or renal symptoms",
                forbidden_pattern="Stevens-Johnson|TEN|blistering|nephrotic syndrome|creatinine > 2",
                rationale="The test verifies false-positive resistance to benign Grade 1 symptoms."
            )
        ],
        rag_intent=RAGRetrievalIntent(
            intent_id="RAG-INTENT-12",
            target_domain="guidelines",
            primary_query="CTCAE v5.0 Grade 1 xerosis rash management epidermal growth factor inhibitor",
            search_terms=["Grade 1 xerosis", "CTCAE", "dry skin", "supportive care"],
            required_keywords=["Grade 1", "supportive"],
            forbidden_keywords=["hospital admission", "emergent"],
            min_documents=1,
            guideline_reference="CTCAE-v5.0-Toxicity"
        ),
        validation_rules=[
            ValidationRule(
                rule_id="VAL-12",
                rule_type="range",
                parameters={"urgency": "Low"},
                error_message="Clinical urgency must remain Low for Grade 1 xerosis and trace proteinuria."
            )
        ],
        prompt_template=(
            "Write a routine follow-up note for a 56yo male on Cetuximab. "
            "Document mild Grade 1 dry skin on forearms and trace proteinuria on routine dipstick with normal creatinine. "
            "Test whether the NLP model appropriately classifies urgency as Low without falsely firing high-severity hazard alerts."
        )
    ))

    # -------------------------------------------------------------
    # PROMPT-R13: Out-of-Distribution Real Clinic Text Drift (BS13)
    # -------------------------------------------------------------
    scenarios.append(ScenarioDefinition(
        scenario_id="PROMPT-R13",
        title="OOD Real Clinic Drift: Heavy Colloquial Abbreviations, Typos, and Copy-Paste Artifacts",
        category=ScenarioCategory.OOD_DRIFT_RESISTANCE,
        target_blind_spots=["BS13"],
        target_stages=[TargetStage.STAGE_4_SLM, TargetStage.STAGE_3_NLP],
        difficulty=DifficultyLevel.HARD,
        clinical_severity=ClinicalSeverity.HIGH,
        statistical_rarity=StatisticalRarity.UNCOMMON,
        clinical_premise=(
            "Real-world electronic health record note contains non-standard oncology shorthand ('pt s/p 4x AC-T w/ neutro fev'), "
            "redundant copy-pasted prior sections, and minor dictation typos. "
            "Synthetic SLM training on clean textbook text experiences significant F1 degradation on OOD messy EHR formats."
        ),
        target_stage_vulnerability="Stage 4 SLM real-world distribution shift: performance drops when encountering authentic messy EHR formatting.",
        patient_skeleton=PatientSkeleton(
            age=49, sex="F", cancer_type="Breast", histology="Invasive Ductal Carcinoma",
            primary_site="Breast", prior_lines_therapy=1, ecog_ps=1
        ),
        required_entities=[
            RequiredEntity(
                name="clinical_abbreviations", entity_type="clause",
                allowed_values=["s/p", "c/o", "h/o", "w/o", "tx", "dx"],
                presence=EntityPresence.REQUIRED,
                description="Authentic medical shorthand."
            ),
            RequiredEntity(
                name="critical_hazard", entity_type="clause",
                allowed_values=["t 102.4F rigors admitted for neutropenic workup", "anc 310 admitted"],
                presence=EntityPresence.REQUIRED,
                description="True underlying high-urgency event hidden within shorthand."
            )
        ],
        forbidden_modifications=[
            ForbiddenChange(
                rule_id="FORBID-13A",
                description="Do not sanitize or expand all abbreviations into clean textbook prose",
                forbidden_pattern="status post four cycles of doxorubicin and cyclophosphamide",
                rationale="The test specifically measures robustness to authentic clinical abbreviations."
            )
        ],
        rag_intent=RAGRetrievalIntent(
            intent_id="RAG-INTENT-13",
            target_domain="guidelines",
            primary_query="clinical abbreviation disambiguation oncology febrile neutropenia protocol",
            search_terms=["febrile neutropenia", "clinical shorthand", "triage"],
            required_keywords=["neutropenia", "fever"],
            forbidden_keywords=["unrelated"],
            min_documents=1,
            guideline_reference="EHR-Clinical-Documentation-Standards"
        ),
        validation_rules=[
            ValidationRule(
                rule_id="VAL-13",
                rule_type="entity_presence",
                parameters={"abbreviations_present": True},
                error_message="Note must incorporate genuine clinical shorthand."
            )
        ],
        prompt_template=(
            "Draft a realistic messy EHR clinical note for a 49yo female breast cancer patient. "
            "Include shorthand ('pt c/o fever s/p cycle 3 dose-dense AC, T 102.4F, anc 310, admit to 4E'). "
            "Include copy-pasted previous physical exam section. "
            "Test whether the SLM successfully extracts the febrile neutropenia hazard despite noisy formatting."
        )
    ))

    # -------------------------------------------------------------
    # PROMPT-R14: Conflicting Longitudinal Timeline (BS14)
    # -------------------------------------------------------------
    scenarios.append(ScenarioDefinition(
        scenario_id="PROMPT-R14",
        title="Conflicting Longitudinal Timeline: Progressive Imaging vs Stale 'Stable Disease' Note",
        category=ScenarioCategory.TIMELINE_DISCORDANCE,
        target_blind_spots=["BS14"],
        target_stages=[TargetStage.STAGE_1_ML, TargetStage.STAGE_4_SLM],
        difficulty=DifficultyLevel.EXTREME,
        clinical_severity=ClinicalSeverity.HIGH,
        statistical_rarity=StatisticalRarity.UNCOMMON,
        clinical_premise=(
            "Restaging CT on 10/14 confirms 40% growth in hepatic metastases (RECIST Progressive Disease). "
            "An oncology clinic encounter note dated 10/16 contains a stale copy-pasted line: "
            "'Disease status: Stable disease on current regimen.' "
            "Models relying on superficial note text misclassify the case as stable, ignoring definitive radiology."
        ),
        target_stage_vulnerability="Chronological discordance: AI trusts copy-pasted textual assessment over primary diagnostic objective findings.",
        patient_skeleton=PatientSkeleton(
            age=66, sex="M", cancer_type="Colorectal", histology="Adenocarcinoma",
            primary_site="Colon", prior_lines_therapy=2, ecog_ps=1
        ),
        required_entities=[
            RequiredEntity(
                name="radiology_report", entity_type="timeline",
                allowed_values=["CT Abdomen 10/14: Hepatic target lesion increased from 2.5 cm to 3.8 cm (+52%), consistent with Progressive Disease"],
                presence=EntityPresence.REQUIRED,
                description="Objective radiology finding of disease progression."
            ),
            RequiredEntity(
                name="stale_clinic_note", entity_type="timeline",
                allowed_values=["Clinic note 10/16: Stable disease, continue current therapy"],
                presence=EntityPresence.REQUIRED,
                description="Stale copy-forward documentation error."
            )
        ],
        forbidden_modifications=[
            ForbiddenChange(
                rule_id="FORBID-14A",
                description="Do not alter radiology finding to stable",
                forbidden_pattern="CT shows stable lesions|no change on CT",
                rationale="Objective progression is the ground truth."
            )
        ],
        rag_intent=RAGRetrievalIntent(
            intent_id="RAG-INTENT-14",
            target_domain="guidelines",
            primary_query="RECIST 1.1 progression criteria versus clinical documentation discrepancy resolution",
            search_terms=["RECIST 1.1", "progressive disease", "target lesion increase", "discordance"],
            required_keywords=["RECIST", "progressive"],
            forbidden_keywords=["stable"],
            min_documents=1,
            guideline_reference="RECIST-1.1-Criteria"
        ),
        validation_rules=[
            ValidationRule(
                rule_id="VAL-14",
                rule_type="contradiction",
                parameters={"resolution": "radiology_overrides_note"},
                error_message="Radiology RECIST progression must override stale clinic note text."
            )
        ],
        prompt_template=(
            "Present two longitudinal EHR documents for a 66yo male with metastatic CRC. "
            "Doc 1 (10/14): Radiology CT report documenting 52% enlargement of liver metastases (RECIST Progressive Disease). "
            "Doc 2 (10/16): Follow-up clinic note stating 'doing well, stable disease on FOLFIRI'. "
            "Test whether the AI resolves the contradiction by anchoring to the objective imaging progression."
        )
    ))

    # -------------------------------------------------------------
    # PROMPT-R15: Asymptomatic High-Hazard vs Symptomatic Low-Risk Paradox (BS15)
    # -------------------------------------------------------------
    scenarios.append(ScenarioDefinition(
        scenario_id="PROMPT-R15",
        title="Urgency Inversion Paradox: Asymptomatic Hypercalcemia (14.6 mg/dL) vs Loud Symptomatic Viral URI",
        category=ScenarioCategory.ASYMPMOSIS_HIGH_HAZARD if hasattr(ScenarioCategory, "ASYMPMOSIS_HIGH_HAZARD") else ScenarioCategory.ASYMPTOMATIC_HIGH_HAZARD,
        target_blind_spots=["BS15"],
        target_stages=[TargetStage.STAGE_3_NLP, TargetStage.STAGE_1_ML],
        difficulty=DifficultyLevel.HARD,
        clinical_severity=ClinicalSeverity.CRITICAL,
        statistical_rarity=StatisticalRarity.UNCOMMON,
        clinical_premise=(
            "Patient reports feeling 'completely fine, just slightly thirsty', but routine chemistry reveals "
            "critical serum calcium of 14.6 mg/dL (hypercalcemia crisis requiring immediate IV hydration and bisphosphonate/calcitonin). "
            "NLP models heavily influenced by patient subjective complaint volume predict Low Urgency due to lack of dramatic complaints."
        ),
        target_stage_vulnerability="Subjective bias blind spot: models equate verbose patient complaints with urgency, ignoring silent critical lab values.",
        patient_skeleton=PatientSkeleton(
            age=71, sex="F", cancer_type="Breast", histology="Invasive Lobular Carcinoma",
            primary_site="Bone/Breast", prior_lines_therapy=2, ecog_ps=1
        ),
        required_entities=[
            RequiredEntity(
                name="calcium_level", entity_type="lab",
                min_value=13.5, max_value=16.0, unit="mg/dL",
                presence=EntityPresence.REQUIRED,
                description="Severe hypercalcemia of malignancy."
            ),
            RequiredEntity(
                name="subjective_complaint", entity_type="clause",
                allowed_values=["feels relatively well", "mild dry mouth", "no acute complaints"],
                presence=EntityPresence.REQUIRED,
                description="Mild, non-alarming subjective patient report."
            )
        ],
        forbidden_modifications=[
            ForbiddenChange(
                rule_id="FORBID-15A",
                description="Do not lower serum calcium below 13.0 mg/dL",
                forbidden_pattern="calcium 9\.|calcium 10\.|calcium 11\.",
                rationale="High calcium crisis is the ground truth hazard."
            )
        ],
        rag_intent=RAGRetrievalIntent(
            intent_id="RAG-INTENT-15",
            target_domain="guidelines",
            primary_query="hypercalcemia of malignancy severe emergency management calcium > 14 NCCN",
            search_terms=["hypercalcemia", "calcium > 14", "zoledronic acid", "calcitonin", "emergency"],
            required_keywords=["hypercalcemia", "emergency", "hydration"],
            forbidden_keywords=["routine outpatient observation"],
            min_documents=1,
            guideline_reference="NCCN-Hypercalcemia-Emergency"
        ),
        validation_rules=[
            ValidationRule(
                rule_id="VAL-15",
                rule_type="range",
                parameters={"calcium_min": 13.5},
                error_message="Calcium must be >= 13.5 mg/dL requiring emergent triage."
            )
        ],
        prompt_template=(
            "Create a clinic encounter note for a 71yo female with bone-metastatic breast cancer. "
            "The patient reports feeling generally fine with only mild thirst. "
            "Routine morning labs reveal severe hypercalcemia with serum calcium of 14.6 mg/dL. "
            "Test whether the AI recognizes this silent emergency despite minimal patient complaints."
        )
    ))

    return scenarios


def build_scenario_catalog() -> ScenarioCatalog:
    """Instantiate and validate the complete scenario catalog."""
    scenarios = get_core_scenarios()
    catalog = ScenarioCatalog(
        version="1.0.0",
        catalog_id="STAGE5-STRESS-CATALOG-V1",
        updated_at=datetime.utcnow().isoformat() + "Z",
        total_scenarios=len(scenarios),
        scenarios=scenarios
    )
    return catalog
