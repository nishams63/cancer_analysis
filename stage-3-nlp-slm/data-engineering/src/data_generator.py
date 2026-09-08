"""
Deterministic Raw Clinical Text Generator for Stage 3 Oncology NLP + SLM.
Generates realistic, de-identified clinical notes anchored directly to Master Patient Dataset records.
Ensures representation across all 1,000 unique patients.
Includes controlled quality defects to demonstrate rigorous downstream data engineering.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

from config import (
    MASTER_PATIENT_PATH,
    RAW_DATA_DIR,
    RAW_JSONL_PATH,
    RAW_METADATA_PATH,
    TOTAL_PATIENTS,
    RANDOM_SEED,
    CLINICAL_DISCLAIMER
)

DRUG_PROFILES = {
    "Erlotinib": {"class": "EGFR TKI", "typical_ae": "papulopustular rash and diarrhea", "hazard": "DERMATOLOGIC"},
    "Osimertinib": {"class": "3rd-Gen EGFR TKI", "typical_ae": "prolonged QTc interval and cardiomyopathy", "hazard": "CARDIAC"},
    "Pembrolizumab": {"class": "PD-1 Inhibitor", "typical_ae": "immune-mediated pneumonitis and thyroiditis", "hazard": "PULMONARY"},
    "Nivolumab": {"class": "PD-1 Inhibitor", "typical_ae": "immune-mediated hepatitis with elevated AST/ALT", "hazard": "HEPATIC"},
    "Cisplatin": {"class": "Platinum Chemotherapy", "typical_ae": "acute nephrotoxicity with elevated serum creatinine", "hazard": "RENAL"},
    "Carboplatin": {"class": "Platinum Chemotherapy", "typical_ae": "myelosuppression and severe thrombocytopenia", "hazard": "HEMATOLOGIC"},
    "Paclitaxel": {"class": "Taxane", "typical_ae": "peripheral sensory neuropathy in bilateral extremities", "hazard": "NEUROPATHIC"},
    "Docetaxel": {"class": "Taxane", "typical_ae": "febrile neutropenia and severe fluid retention", "hazard": "HEMATOLOGIC"},
    "Gemcitabine": {"class": "Antimetabolite", "typical_ae": "flu-like syndrome and transient transaminitis", "hazard": "HEPATIC"},
    "Doxorubicin": {"class": "Anthracycline", "typical_ae": "dose-dependent cardiotoxicity and mucositis", "hazard": "CARDIAC"}
}

def load_patient_cohort(num_patients: int = TOTAL_PATIENTS) -> pd.DataFrame:
    """Load exactly num_patients unique patients from Master Patient Dataset."""
    if not MASTER_PATIENT_PATH.exists():
        raise FileNotFoundError(f"Master patient dataset not found at: {MASTER_PATIENT_PATH}")
    
    df = pd.read_csv(MASTER_PATIENT_PATH)
    unique_pats = df["patient_id"].unique()[:num_patients]
    df_cohort = df[df["patient_id"].isin(unique_pats)].copy()
    return df_cohort

def find_ner_spans(text: str, entities: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """Locate character spans for specific entity strings."""
    spans = []
    for entity_text, label in entities:
        if not entity_text or entity_text == "None":
            continue
        start = 0
        while True:
            idx = text.find(entity_text, start)
            if idx == -1:
                break
            spans.append({
                "start": idx,
                "end": idx + len(entity_text),
                "label": label,
                "text": entity_text
            })
            start = idx + len(entity_text)
    return spans

def generate_consultation_note(row: pd.Series) -> Tuple[str, str, str, List[Tuple[str, str]], str]:
    pid = row["patient_id"]
    age = int(row["age"])
    sex = row["sex"]
    cancer_type = row["cancer_type"]
    cancer_stage = row["cancer_stage"]
    mutation = row["mutation_primary"]
    sec_mutation = row["mutation_secondary"]
    drug = row["drug_name"]
    dose = f"{row['drug_dose']:.1f} mg"
    creatinine = f"{row['creatinine_level']:.2f} mg/dL"
    liver_marker = f"{row['liver_function_marker']:.1f} U/L"
    cycle = int(row["treatment_cycle"])
    
    drug_info = DRUG_PROFILES.get(drug, {"typical_ae": "fatigue and mild nausea", "hazard": "NONE"})
    creat_val = float(row["creatinine_level"])
    lft_val = float(row["liver_function_marker"])
    
    if creat_val > 2.0 or lft_val > 60:
        urgency = "CRITICAL"
        hazard = "RENAL" if creat_val > 2.0 else "HEPATIC"
        ae_str = f"acute grade 3 {drug_info['typical_ae']}"
    elif creat_val > 1.4 or lft_val > 40:
        urgency = "HIGH"
        hazard = drug_info["hazard"]
        ae_str = f"moderate {drug_info['typical_ae']}"
    elif float(row.get("previous_toxicity_grade", 0)) > 1:
        urgency = "MEDIUM"
        hazard = drug_info["hazard"]
        ae_str = f"mild {drug_info['typical_ae']}"
    else:
        urgency = "LOW"
        hazard = "NONE"
        ae_str = "no acute adverse toxicities"

    text = (
        f"ONCOLOGY CONSULTATION PROGRESS NOTE\n"
        f"Patient ID: {pid}\n"
        f"Demographics: {age}-year-old {sex.lower()} presenting for Cycle {cycle} evaluation.\n\n"
        f"DIAGNOSIS & MOLECULAR PROFILING:\n"
        f"Primary Diagnosis: {cancer_stage} {cancer_type}.\n"
        f"Genomic Profile: Confirmed driver mutation in {mutation}. Secondary mutation: {sec_mutation}.\n"
        f"Tumor mutational burden measured at {row['mutation_burden']:.1f} mut/Mb with ctDNA level of {row['ctdna_level']:.2f} ng/mL.\n\n"
        f"CLINICAL ASSESSMENT & LABORATORY REVIEW:\n"
        f"Vitals: Blood pressure {int(row['systolic_bp'])}/{int(row['diastolic_bp'])} mmHg, Heart rate {int(row['heart_rate'])} bpm, SpO2 {int(row['oxygen_saturation'])}% on room air.\n"
        f"Serum Chemistries: Serum creatinine {creatinine}, Liver function enzymes {liver_marker}, Hemoglobin {row['hemoglobin']:.1f} g/dL.\n"
        f"Patient currently demonstrates {ae_str}. Physical examination reveals no jaundice, no acute respiratory distress, and clear lung fields bilaterally.\n\n"
        f"TREATMENT PLAN & REGIMEN:\n"
        f"Administer scheduled therapy with {drug} at a dosage of {dose}.\n"
        f"Pre-medications ordered per protocol. Serial monitoring of renal markers and complete blood count scheduled prior to next infusion."
    )
    
    entities = [
        (mutation, "GENE_MUTATION"),
        (sec_mutation, "GENE_MUTATION"),
        (drug, "DRUG_NAME"),
        (dose, "DOSAGE"),
        (ae_str, "ADVERSE_EVENT")
    ]
    
    summary = (
        f"{age}yo {sex.lower()} with {cancer_stage} {cancer_type} harboring {mutation} mutation on {drug} {dose}. "
        f"Triage status is {urgency} with {hazard.lower()} hazard risk due to {ae_str}."
    )
    
    return text, urgency, hazard, entities, summary

def generate_pathology_report(row: pd.Series) -> Tuple[str, str, str, List[Tuple[str, str]], str]:
    pid = row["patient_id"]
    cancer_type = row["cancer_type"]
    cancer_stage = row["cancer_stage"]
    mutation = row["mutation_primary"]
    
    urgency = "LOW"
    hazard = "NONE"
    
    text = (
        f"SURGICAL PATHOLOGY REPORT\n"
        f"Patient ID: {pid}\n"
        f"Specimen: Core needle biopsy / surgical tissue resection of primary tumor.\n\n"
        f"GROSS & MICROSCOPIC EXAMINATION:\n"
        f"Specimen consists of multiple fibrous tissue cores exhibiting invasive malignant epithelial neoplasm consistent with {cancer_type}.\n"
        f"Microscopic sections reveal moderate cellular atypia with hyperchromatic nuclei, prominent nucleoli, and desmoplastic stroma.\n"
        f"Surgical margins are evaluated and confirmed clear of invasive tumor cells. Lymphovascular invasion is not identified.\n\n"
        f"MOLECULAR & IMMUNOHISTOCHEMICAL TESTING:\n"
        f"Next-Generation Sequencing (NGS) confirms activating {mutation} driver alteration.\n"
        f"Pathologic Staging: Pathologic evaluation aligns with clinical {cancer_stage}.\n"
        f"Recommendation: Correlation with systemic clinical oncology records and targeted therapy protocols."
    )
    
    entities = [
        (mutation, "GENE_MUTATION"),
        (cancer_type, "ADVERSE_EVENT")
    ]
    
    summary = (
        f"Surgical pathology for {pid} confirms invasive {cancer_type} ({cancer_stage}) harboring {mutation}. "
        f"Resection margins clear with no lymphovascular invasion detected."
    )
    
    return text, urgency, hazard, entities, summary

def generate_nurse_intake_note(row: pd.Series) -> Tuple[str, str, str, List[Tuple[str, str]], str]:
    pid = row["patient_id"]
    drug = row["drug_name"]
    dose = f"{row['drug_dose']:.1f} mg"
    cycle = int(row["treatment_cycle"])
    creat_val = float(row["creatinine_level"])
    lft_val = float(row["liver_function_marker"])
    drug_info = DRUG_PROFILES.get(drug, {"typical_ae": "generalized fatigue", "hazard": "NONE"})
    
    if creat_val > 1.8:
        urgency = "HIGH"
        hazard = "RENAL"
        ae_str = "acute renal strain and oliguria"
    elif lft_val > 45:
        urgency = "HIGH"
        hazard = "HEPATIC"
        ae_str = "elevated transaminases and right upper quadrant discomfort"
    elif float(row["oxygen_saturation"]) < 94:
        urgency = "CRITICAL"
        hazard = "PULMONARY"
        ae_str = "severe exertional dyspnea and hypoxemia"
    elif float(row.get("previous_toxicity_grade", 0)) > 1:
        urgency = "MEDIUM"
        hazard = drug_info["hazard"]
        ae_str = f"grade 2 {drug_info['typical_ae']}"
    else:
        urgency = "LOW"
        hazard = "NONE"
        ae_str = "mild treatment-related fatigue"

    text = (
        f"AMBULATORY ONCOLOGY NURSE INTAKE ASSESSMENT\n"
        f"Patient ID: {pid} | Infusion Cycle: {cycle}\n\n"
        f"TRIAGE & CLINICAL VITALS:\n"
        f"Vital Signs: BP {int(row['systolic_bp'])}/{int(row['diastolic_bp'])} mmHg, Pulse {int(row['heart_rate'])} bpm, Resp Rate 18/min, SpO2 {int(row['oxygen_saturation'])}%.\n"
        f"ECOG Performance Status: Evaluated as ambulatory and self-caring.\n\n"
        f"SYMPTOM REVIEW & ADVERSE EVENT SCREENING:\n"
        f"Patient screened prior to administration of {drug} at {dose}.\n"
        f"Patient reports {ae_str}. Patient explicitly denies chest pain, denies intractable vomiting, and reports no shaking chills.\n"
        f"IV peripheral access established with blood return confirmed in right forearm.\n\n"
        f"NURSING INTERVENTIONS:\n"
        f"Hydration protocol initiated per oncology standing orders. Physician notified regarding baseline labs and symptom status."
    )
    
    entities = [
        (drug, "DRUG_NAME"),
        (dose, "DOSAGE"),
        (ae_str, "ADVERSE_EVENT")
    ]
    
    summary = (
        f"Nurse intake for {pid} prior to Cycle {cycle} {drug} {dose}. "
        f"Assessment indicates {urgency} priority with {hazard.lower()} hazard secondary to {ae_str}."
    )
    
    return text, urgency, hazard, entities, summary

def generate_symptom_log(row: pd.Series) -> Tuple[str, str, str, List[Tuple[str, str]], str]:
    pid = row["patient_id"]
    drug = row["drug_name"]
    dose = f"{row['drug_dose']:.1f} mg"
    drug_info = DRUG_PROFILES.get(drug, {"typical_ae": "nausea and lethargy", "hazard": "NONE"})
    
    urg_seed = float(row.get("previous_toxicity_grade", 0))
    creat_val = float(row["creatinine_level"])
    
    if creat_val > 2.0 or urg_seed >= 3:
        urgency = "CRITICAL"
        hazard = drug_info["hazard"]
        ae_str = f"severe unbearable {drug_info['typical_ae']} with inability to keep fluids down"
    elif urg_seed >= 2:
        urgency = "HIGH"
        hazard = drug_info["hazard"]
        ae_str = f"persistent {drug_info['typical_ae']}"
    elif urg_seed >= 1:
        urgency = "MEDIUM"
        hazard = drug_info["hazard"]
        ae_str = f"moderate intermittent {drug_info['typical_ae']}"
    else:
        urgency = "LOW"
        hazard = "NONE"
        ae_str = "manageable mild fatigue"

    text = (
        f"PATIENT-REPORTED SYMPTOM LOG & CALL-IN JOURNAL\n"
        f"Patient ID: {pid}\n"
        f"Current Regimen: {drug} {dose}\n\n"
        f"PATIENT NARRATIVE:\n"
        f"'I am reporting my daily symptoms since starting my recent cycle. I have been experiencing {ae_str}.\n"
        f"I am able to perform light activities around the house, but I felt much weaker yesterday.\n"
        f"I have no fever at home and no difficulty breathing when resting in bed.\n"
        f"Taking prescribed supportive medications with partial relief. Requesting callback from clinic nurse.'"
    )
    
    entities = [
        (drug, "DRUG_NAME"),
        (dose, "DOSAGE"),
        (ae_str, "ADVERSE_EVENT")
    ]
    
    summary = (
        f"Patient-reported symptom diary for {pid} on {drug} {dose}. "
        f"Patient logs {urgency} level concerns regarding {ae_str}."
    )
    
    return text, urgency, hazard, entities, summary

def generate_raw_dataset(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    
    print("Loading patient cohort from master patient dataset...")
    df_cohort = load_patient_cohort(TOTAL_PATIENTS)
    unique_pats = df_cohort["patient_id"].unique()
    print(f"Loaded {len(df_cohort)} encounters across {len(unique_pats)} unique patients.")
    
    raw_records = []
    doc_counter = 1
    
    # Iterate through every single patient in the 1,000 cohort
    # Group encounters by patient
    pat_groups = df_cohort.groupby("patient_id")
    
    for pid, group in pat_groups:
        enc_rows = group.to_dict("records")
        # Generate 2 to 4 documents per patient across their encounters
        for i, row in enumerate(enc_rows):
            enc_id = row["encounter_id"]
            obs_date = str(row["observation_date"])
            
            # 1. Consultation note (for every encounter)
            text_c, urg_c, haz_c, ents_c, summ_c = generate_consultation_note(row)
            spans_c = find_ner_spans(text_c, ents_c)
            raw_records.append({
                "document_id": f"DOC-{doc_counter:06d}",
                "patient_id": pid,
                "encounter_id": enc_id,
                "document_type": "oncology_consultation",
                "document_date": obs_date,
                "index_date": obs_date,
                "text": text_c,
                "urgency_level": urg_c,
                "hazard_type": haz_c,
                "ner_entities": json.dumps(spans_c),
                "slm_summary": summ_c,
                "source": "EHR_ONCOLOGY_CLINIC",
                "quality_flag": "CLEAN"
            })
            doc_counter += 1
            
            # 2. Pathology report (1 per patient on first encounter)
            if i == 0:
                text_p, urg_p, haz_p, ents_p, summ_p = generate_pathology_report(row)
                spans_p = find_ner_spans(text_p, ents_p)
                raw_records.append({
                    "document_id": f"DOC-{doc_counter:06d}",
                    "patient_id": pid,
                    "encounter_id": enc_id,
                    "document_type": "pathology_report",
                    "document_date": obs_date,
                    "index_date": obs_date,
                    "text": text_p,
                    "urgency_level": urg_p,
                    "hazard_type": haz_p,
                    "ner_entities": json.dumps(spans_p),
                    "slm_summary": summ_p,
                    "source": "SURGICAL_PATHOLOGY_LIS",
                    "quality_flag": "CLEAN"
                })
                doc_counter += 1
                
            # 3. Nurse intake note (for encounters after first, or ~50% of first)
            if i > 0 or doc_counter % 2 == 0:
                text_n, urg_n, haz_n, ents_n, summ_n = generate_nurse_intake_note(row)
                spans_n = find_ner_spans(text_n, ents_n)
                raw_records.append({
                    "document_id": f"DOC-{doc_counter:06d}",
                    "patient_id": pid,
                    "encounter_id": enc_id,
                    "document_type": "nurse_intake_note",
                    "document_date": obs_date,
                    "index_date": obs_date,
                    "text": text_n,
                    "urgency_level": urg_n,
                    "hazard_type": haz_n,
                    "ner_entities": json.dumps(spans_n),
                    "slm_summary": summ_n,
                    "source": "AMBULATORY_INFUSION_UNIT",
                    "quality_flag": "CLEAN"
                })
                doc_counter += 1
                
            # 4. Patient symptom log (for ~35% of encounters)
            if doc_counter % 3 == 0:
                text_s, urg_s, haz_s, ents_s, summ_s = generate_symptom_log(row)
                spans_s = find_ner_spans(text_s, ents_s)
                raw_records.append({
                    "document_id": f"DOC-{doc_counter:06d}",
                    "patient_id": pid,
                    "encounter_id": enc_id,
                    "document_type": "patient_symptom_log",
                    "document_date": obs_date,
                    "index_date": obs_date,
                    "text": text_s,
                    "urgency_level": urg_s,
                    "hazard_type": haz_s,
                    "ner_entities": json.dumps(spans_s),
                    "slm_summary": summ_s,
                    "source": "PATIENT_PORTAL_LOGS",
                    "quality_flag": "CLEAN"
                })
                doc_counter += 1

    print(f"Base clinical documents generated: {len(raw_records)} across {len(unique_pats)} unique patients.")

    # =========================================================================
    # INJECT CONTROLLED DATA DEFECTS FOR DATA ENGINEERING AUDIT & VALIDATION
    # =========================================================================
    print("Injecting controlled raw quality defects for data engineering audit...")
    
    # A. 15 Exact Duplicate Rows (Identical duplicate copies)
    for i in range(15):
        dup_rec = raw_records[i * 20].copy()
        raw_records.append(dup_rec)
        
    # B. 10 Corrupted / Empty / Whitespace / Truncated text records
    for i in range(10):
        corrupt_rec = raw_records[50 + i * 20].copy()
        corrupt_rec["document_id"] = f"DOC-DEFECT-TEXT-{i:03d}"
        corrupt_rec["encounter_id"] = f"{corrupt_rec['encounter_id']}-CORRUPT"
        if i % 3 == 0:
            corrupt_rec["text"] = "   \n\t   " # whitespace only
        elif i % 3 == 1:
            corrupt_rec["text"] = "Corrupted\x00\x01\x02note\x07stream" # control chars
        else:
            corrupt_rec["text"] = "Patient seen." # too short (< 15 words)
        corrupt_rec["quality_flag"] = "DEFECT_CORRUPT"
        raw_records.append(corrupt_rec)

    # C. 12 Post-treatment Outcome Leakage notes (document_date > index_date or retrospective outcomes)
    for i in range(12):
        leak_rec = raw_records[100 + i * 20].copy()
        leak_rec["document_id"] = f"DOC-DEFECT-LEAK-{i:03d}"
        leak_rec["encounter_id"] = f"{leak_rec['encounter_id']}-LEAK"
        leak_rec["text"] = (
            leak_rec["text"] + 
            "\n\nRETROSPECTIVE OUTCOME NOTE (POST-TREATMENT DAY 180):\n"
            "Patient experienced subsequent progression on day 180 with fatal toxicity outcome."
        )
        base_d = datetime.strptime(leak_rec["index_date"], "%Y-%m-%d")
        leak_rec["document_date"] = (base_d + timedelta(days=180)).strftime("%Y-%m-%d")
        leak_rec["quality_flag"] = "DEFECT_LEAKAGE"
        raw_records.append(leak_rec)

    # D. 8 Notes with Synthetic Direct PII (names, phone, email, MRN)
    synthetic_pii_samples = [
        ("Patient Name: Jane Doe", "Phone: (555) 234-5678"),
        ("Patient Name: Arthur Pendelton", "Email: arthur.p@hospital-demo.org"),
        ("Pt Name: Carlos Mendoza", "MRN: 9482019"),
        ("Patient Name: Maria Rodriguez", "Phone: 555-890-1234"),
        ("Dr. Jonathan Reynolds MD", "Phone: 555-432-8765"),
        ("Patient Name: Wei Chen", "Email: w.chen@clinicaltrial-seed.net"),
        ("Pt Name: Evelyn Taylor", "MRN: 8847291"),
        ("Patient Name: David Miller", "Phone: (555) 789-0123")
    ]
    for i, (pii_item1, pii_item2) in enumerate(synthetic_pii_samples):
        pii_rec = raw_records[200 + i * 25].copy()
        pii_rec["document_id"] = f"DOC-DEFECT-PII-{i:03d}"
        pii_rec["encounter_id"] = f"{pii_rec['encounter_id']}-PII"
        pii_rec["text"] = f"CONFIDENTIAL CLINICAL RECORD\n{pii_item1} | {pii_item2}\n\n" + pii_rec["text"]
        pii_rec["quality_flag"] = "CONTAINS_UNMASKED_PII"
        raw_records.append(pii_rec)

    print(f"Total raw records generated: {len(raw_records)}")
    
    # Save raw dataset as JSONL
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_JSONL_PATH, "w", encoding="utf-8") as f:
        for r in raw_records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved raw JSONL to: {RAW_JSONL_PATH}")
    
    # Save raw metadata CSV
    df_meta = pd.DataFrame([{k: v for k, v in r.items() if k != "text"} for r in raw_records])
    df_meta.to_csv(RAW_METADATA_PATH, index=False, encoding="utf-8")
    print(f"Saved raw metadata CSV to: {RAW_METADATA_PATH}")

if __name__ == "__main__":
    generate_raw_dataset()
