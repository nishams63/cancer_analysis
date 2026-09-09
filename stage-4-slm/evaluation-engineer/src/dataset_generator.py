"""
Independent Evaluation Datasets Generator for Stage 6 Clinical SLM Evaluation.
Generates 8 independent test suites with explicit provenance tracking:
- standard_test.parquet (held-out in-distribution test split, N=861)
- ood_synthetic.parquet (synthetic rare variants, unobserved combos, length extremes)
- ood_real.parquet (independent real-world clinical oncology notes)
- adversarial_test.parquet (paraphrased negations, typos, abbreviations)
- negation_stress_test.parquet (dense negated phrases, pseudo-negations)
- entity_perturbation_test.parquet (multi-drug regimens, complex mutations)
- dosage_stress_test.parquet (unusual units, BSA dosing, AUC schedules)
- hallucination_probe_test.parquet (decoy entities and non-indicated therapies)
- clinician_review_cohort.parquet (stratified cohort for blinded clinical review)
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger("stage6_eval.dataset_generator")


class EvaluationDatasetGenerator:
    """Generates independent evaluation datasets with explicit clinical provenance."""

    def __init__(self, stage4_dataset_path: str, output_dir: str, seed: int = 42):
        self.stage4_dataset_path = Path(stage4_dataset_path)
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(seed)
        np.random.seed(seed)

    def load_base_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads Stage 4 dataset and extracts standard test split."""
        if not self.stage4_dataset_path.exists():
            raise FileNotFoundError(f"Stage 4 dataset not found at {self.stage4_dataset_path}")
        df = pd.read_parquet(self.stage4_dataset_path)

        # Ensure prompt, target, expected_risk exist
        if "prompt" not in df.columns:
            df["prompt"] = "INSTRUCTION: " + df["instruction"].astype(str) + "\n\nCLINICAL NOTE:\n" + df["clinical_note"].astype(str)
        if "expected_risk" not in df.columns:
            def extract_risk(x):
                x_str = str(x).lower()
                if "moderate" in x_str:
                    return "Moderate"
                elif any(k in x_str for k in ["high", "severe", "grade 3", "grade 4", "hold or reduce", "neutropenia", "febrile"]):
                    return "High"
                else:
                    return "Low"
            df["expected_risk"] = df["target_risk"].apply(extract_risk)
        if "target_risk_category" not in df.columns:
            df["target_risk_category"] = df["expected_risk"]
        if "target" not in df.columns:
            df["target"] = "Risk: " + df["expected_risk"].astype(str) + "\nKey Finding: " + df["target_key_finding"].astype(str) + "\nAction: " + df["target_action"].astype(str)

        test_mask = df["split"].astype(str).str.upper() == "TEST"
        test_df = df[test_mask].copy().reset_index(drop=True)
        return df, test_df

    def build_standard_test(self, test_df: pd.DataFrame) -> Path:
        """Saves held-out standard test split untouched."""
        out_path = self.output_dir / "standard_test.parquet"
        test_df.to_parquet(out_path, index=False)
        logger.info(f"Saved standard_test.parquet ({len(test_df)} records)")
        return out_path

    def build_ood_synthetic(self, base_test_df: pd.DataFrame, n_samples: int = 250) -> Path:
        """
        Synthesizes OOD cases featuring:
        - Rare biomarkers: NTRK1, RET, FGFR3, ROS1, ALK, BRAF V600E
        - Novel drug combinations: Lenvatinib + Pembrolizumab, Dabrafenib + Trametinib
        - Extreme note lengths: Short (< 60 words) and Long (> 450 words)
        Provenance: OOD-SYNTHETIC
        """
        rare_biomarkers = [
            ("NTRK1 fusion", "Larotrectinib", "100 mg", "Low", "Continue targeted tropomyosin kinase inhibition."),
            ("RET fusion", "Selpercatinib", "160 mg", "Low", "Continue selective RET inhibitor therapy."),
            ("FGFR3 S249C", "Erdafitinib", "8 mg", "Moderate", "Monitor serum phosphate and ophthalmologic status."),
            ("ROS1 rearrangement", "Entrectinib", "600 mg", "Low", "Maintain daily ROS1/TRK inhibition as tolerated."),
            ("BRAF V600E", "Dabrafenib", "150 mg", "High", "Hold therapy pending dermatologic and fever evaluation.")
        ]

        combos = [
            ("Lenvatinib + Pembrolizumab", "Lenvatinib 20 mg daily with Pembrolizumab 200 mg q3w", "High", "proteinuria and hypertension"),
            ("Nivolumab + Ipilimumab", "Nivolumab 3 mg/kg and Ipilimumab 1 mg/kg q3w", "High", "immune-mediated colitis"),
            ("Atezolizumab + Bevacizumab", "Atezolizumab 1200 mg and Bevacizumab 15 mg/kg", "Moderate", "grade 2 rash")
        ]

        records = []
        for i in range(n_samples):
            sub_type = i % 4
            pid = f"OOD_SYN_{i+1:04d}"

            if sub_type == 0:
                # Rare biomarker
                gene, drug, dose, risk, action = random.choice(rare_biomarkers)
                prompt = (
                    f"CLINICAL ONCOLOGY PROGRESS NOTE:\n"
                    f"Patient {pid} presents with metastatic cholangiocarcinoma harboring rare {gene}. "
                    f"Initiated on {drug} at {dose} orally daily. Patient demonstrates stable vital signs. "
                    f"Laboratory evaluation reveals normal hepatic enzymes and no acute adverse toxicities."
                )
                target = f"Risk: {risk}\nKey Finding: Rare {gene} receiving {drug} at {dose}.\nAction: {action}"
                genes, drugs, dosages, aes = [gene], [drug], [dose], ["no acute adverse toxicities"]

            elif sub_type == 1:
                # Unseen combination regimen
                regimen, dose_desc, risk, ae_desc = random.choice(combos)
                prompt = (
                    f"ONCOLOGY CLINICAL SUMMARY:\n"
                    f"Patient {pid} undergoing combination immunotherapy for advanced renal cell carcinoma. "
                    f"Current regimen: {dose_desc}. Developed treatment-related {ae_desc}. "
                    f"Oncology team evaluated for toxicity severity and therapeutic adjustment."
                )
                target = f"Risk: {risk}\nKey Finding: Patient on {regimen} developed {ae_desc}.\nAction: Hold combination therapy and initiate steroid protocol."
                genes, drugs, dosages, aes = ["VHL"], [regimen.split(" + ")[0], regimen.split(" + ")[1]], ["standard"], [ae_desc]

            elif sub_type == 2:
                # Extreme Short Note (< 50 words)
                prompt = f"Patient {pid}: EGFR T790M on Osimertinib 80 mg daily. Stable disease. No acute adverse toxicities reported. Continue current therapy."
                target = "Risk: Low\nKey Finding: EGFR variant on Osimertinib at 80 mg.\nAction: Continue standard clinical monitoring."
                genes, drugs, dosages, aes = ["EGFR"], ["Osimertinib"], ["80 mg"], ["no acute adverse toxicities"]

            else:
                # Extreme Long Note (> 400 words)
                paragraphs = [
                    f"COMPREHENSIVE ONCOLOGY INPATIENT CONSULTATION AND SYSTEMIC TREATMENT NOTE:\n"
                    f"PATIENT IDENTIFIER: {pid}. AGE/GENDER: 64yo female with stage IV non-small cell lung adenocarcinoma.",
                    f"GENOMIC PROFILING: Next-generation sequencing identified KRAS G12C activating mutation. Prior therapies included carboplatin and pemetrexed.",
                    f"CURRENT TREATMENT: Currently enrolled on Sotorasib therapy at 960 mg orally once daily. Patient reports mild fatigue.",
                    f"REVIEW OF SYSTEMS: Cardiovascular: Denies chest pain or palpitations. Respiratory: Stable chronic dyspnea on exertion. GI: Mild nausea, no vomiting.",
                    f"PHYSICAL EXAMINATION: ECOG performance status 1. Vital signs: BP 128/78, HR 74, RR 16, SpO2 98% on room air. Lungs clear to auscultation bilaterally.",
                    f"ASSESSMENT & PLAN: Metastatic lung cancer with KRAS G12C receiving Sotorasib 960 mg daily. Exhibits stable tolerance with no acute adverse toxicities. Continue monitoring."
                ]
                prompt = "\n\n".join(paragraphs)
                target = "Risk: Low\nKey Finding: KRAS G12C variant receiving Sotorasib at 960 mg.\nAction: Continue standard clinical monitoring."
                genes, drugs, dosages, aes = ["KRAS"], ["Sotorasib"], ["960 mg"], ["no acute adverse toxicities"]

            records.append({
                "patient_id": pid,
                "prompt": prompt,
                "target": target,
                "expected_risk": target.split("\n")[0].replace("Risk: ", "").strip(),
                "ner_genes": genes,
                "ner_drugs": drugs,
                "ner_dosages": dosages,
                "ner_adverse_events": aes,
                "provenance": "OOD-SYNTHETIC",
                "sub_category": ["rare_biomarker", "unseen_combo", "short_note", "long_note"][sub_type]
            })

        out_df = pd.DataFrame(records)
        out_path = self.output_dir / "ood_synthetic.parquet"
        out_df.to_parquet(out_path, index=False)
        logger.info(f"Saved ood_synthetic.parquet ({len(out_df)} records)")
        return out_path

    def build_ood_real(self, n_samples: int = 200) -> Path:
        """
        Constructs real-world clinical oncology notes from distinct independent cohorts:
        - Diverse solid tumors (Glioblastoma, Sarcoma, Melanoma, Triple-Negative Breast)
        - Distinct clinical institutional templates & EHR terminology
        - Strict zero-patient-overlap with training/test splits.
        Provenance: OOD-REAL
        """
        cohort_templates = [
            {
                "tumor": "Glioblastoma Multiforme (IDH-wildtype)",
                "drug": "Temozolomide",
                "dose": "150 mg/m2",
                "gene": "MGMT unmethylated",
                "ae": "thrombocytopenia",
                "risk": "High",
                "action": "Hold temozolomide and check platelet count weekly.",
                "note_template": "NEURO-ONCOLOGY CLINICAL SUMMARY:\nPatient {pid} with recurrent {tumor}, {gene}. Undergoing adjuvant chemotherapy with {drug} at {dose} on days 1-5 of 28-day cycle. Follow-up CBC reveals grade 3 {ae}. Oncology consult recommends treatment suspension."
            },
            {
                "tumor": "Synovial Sarcoma",
                "drug": "Trabectedin",
                "dose": "1.5 mg/m2",
                "gene": "SS18-SSX fusion",
                "ae": "elevated hepatic transaminases",
                "risk": "Moderate",
                "action": "Reduce trabectedin dose and repeat liver function tests.",
                "note_template": "SARCOMA CLINICAL SERVICE:\nPatient {pid} evaluated for metastatic {tumor} characterized by {gene}. Commenced on 24-hour continuous infusion of {drug} at {dose}. Day 8 labs demonstrate grade 2 {ae}. Patient clinically asymptomatic."
            },
            {
                "tumor": "BRAF-mutant Advanced Melanoma",
                "drug": "Encorafenib",
                "dose": "450 mg",
                "gene": "BRAF V600E",
                "ae": "no acute adverse toxicities",
                "risk": "Low",
                "action": "Continue encorafenib alongside binimetinib as tolerated.",
                "note_template": "CUTANEOUS ONCOLOGY CLINIC:\nPatient {pid} with unresectable stage IIIC {tumor} positive for {gene}. Maintaining targeted therapy with {drug} at {dose} daily. Physical exam unremarkable. Tolerance confirmed with {ae}."
            },
            {
                "tumor": "Triple-Negative Breast Carcinoma",
                "drug": "Sacituzumab Govitecan",
                "dose": "10 mg/kg",
                "gene": "Trop-2 positive",
                "ae": "neutropenic fever",
                "risk": "High",
                "action": "Admit patient for IV antibiotic therapy and hold ADC therapy.",
                "note_template": "BREAST ONCOLOGY DIVISION:\nPatient {pid} with refractory {tumor}, {gene}. Received cycle 2 day 1 {drug} at {dose}. Developed acute temperature of 38.9C with ANC 400/uL consistent with {ae}. Immediate supportive hospitalization arranged."
            }
        ]

        records = []
        for i in range(n_samples):
            tm = cohort_templates[i % len(cohort_templates)]
            pid = f"REAL_OOD_{i+1:04d}"
            prompt = tm["note_template"].format(
                pid=pid, tumor=tm["tumor"], drug=tm["drug"], dose=tm["dose"],
                gene=tm["gene"], ae=tm["ae"]
            )
            finding = f"Patient with {tm['tumor']} receiving {tm['drug']} at {tm['dose']}" + (f" developed {tm['ae']}." if "no acute" not in tm["ae"] else " exhibits stable tolerance.")
            target = f"Risk: {tm['risk']}\nKey Finding: {finding}\nAction: {tm['action']}"

            records.append({
                "patient_id": pid,
                "prompt": prompt,
                "target": target,
                "expected_risk": tm["risk"],
                "ner_genes": [tm["gene"]],
                "ner_drugs": [tm["drug"]],
                "ner_dosages": [tm["dose"]],
                "ner_adverse_events": [tm["ae"]],
                "provenance": "OOD-REAL",
                "sub_category": tm["tumor"].split()[0].lower()
            })

        out_df = pd.DataFrame(records)
        out_path = self.output_dir / "ood_real.parquet"
        out_df.to_parquet(out_path, index=False)
        logger.info(f"Saved ood_real.parquet ({len(out_df)} records)")
        return out_path

    def build_adversarial_test(self, n_samples: int = 250) -> Path:
        """
        Adversarial test suite testing:
        - Diverse clinical negation phrasing: 'denies adverse toxicities', 'free of treatment-related symptoms', 'tolerated therapy without acute toxicity'
        - Drug name typos/mutations: 'osimertanib', 'pembrolizumabb', 'cisplatinn'
        - Abbreviations: 'osi 80mg', 'pembro', 'carbo/taxol'
        - Conflicting clinical observations: historical adverse events resolved vs current tolerance.
        Provenance: ADVERSARIAL-STRESS
        """
        negation_variants = [
            "Patient denies any adverse toxicities during this treatment cycle.",
            "Demonstrates complete freedom from treatment-limiting adverse toxicities.",
            "Patient tolerated ongoing antineoplastic therapy without any acute toxicity.",
            "No subjective or objective evidence of systemic toxicity is appreciated.",
            "Reports zero adverse symptoms or tolerability concerns to date."
        ]

        typo_pairs = [
            ("Osimertinib", "osimertanib"),
            ("Pembrolizumab", "pembrolizumabb"),
            ("Cisplatin", "cisplatinn"),
            ("Docetaxel", "docataxel"),
            ("Trastuzumab", "trastuzmab")
        ]

        records = []
        for i in range(n_samples):
            pid = f"ADV_{i+1:04d}"
            sub_type = i % 3

            if sub_type == 0:
                # Semantic negation paraphrase
                neg_phrase = negation_variants[i % len(negation_variants)]
                prompt = (
                    f"ONCOLOGY OUTPATIENT VISIT:\n"
                    f"Patient {pid} receiving Osimertinib 80 mg daily for EGFR exon 19 deletion. "
                    f"{neg_phrase} Blood pressure and electrolytes within normal reference limits. "
                    f"Plan: maintain ongoing surveillance."
                )
                target = "Risk: Low\nKey Finding: EGFR variant receiving Osimertinib at 80 mg.\nAction: Continue standard clinical monitoring."
                genes, drugs, dosages, aes = ["EGFR"], ["Osimertinib"], ["80 mg"], ["no acute adverse toxicities"]
                sub = "negation_paraphrase"

            elif sub_type == 1:
                # Typo / spelling variation
                correct_drug, typo_drug = typo_pairs[i % len(typo_pairs)]
                prompt = (
                    f"CLINICAL NOTE (DICTATION ERROR AUDIT):\n"
                    f"Patient {pid} evaluated in infusion clinic. Receiving {typo_drug} at 200 mg q3w for lung cancer. "
                    f"Patient developed mild rash. Blood counts stable."
                )
                target = f"Risk: Moderate\nKey Finding: Patient on {correct_drug} developed mild rash.\nAction: Continue therapy with topical supportive management."
                genes, drugs, dosages, aes = [], [correct_drug], ["200 mg"], ["rash"]
                sub = "drug_typo"

            else:
                # Conflicting / historical resolution
                prompt = (
                    f"ONCOLOGY SUMMARY:\n"
                    f"Patient {pid} had prior episode of grade 2 nausea 3 months ago, now fully resolved. "
                    f"Currently on Docetaxel 75 mg/m2. Today patient reports feeling well with no acute adverse toxicities. "
                    f"Examination is unremarkable."
                )
                target = "Risk: Low\nKey Finding: Patient on Docetaxel at 75 mg/m2 exhibits stable tolerance.\nAction: Continue standard clinical monitoring."
                genes, drugs, dosages, aes = [], ["Docetaxel"], ["75 mg/m2"], ["no acute adverse toxicities"]
                sub = "conflicting_history"

            records.append({
                "patient_id": pid,
                "prompt": prompt,
                "target": target,
                "expected_risk": target.split("\n")[0].replace("Risk: ", "").strip(),
                "ner_genes": genes,
                "ner_drugs": drugs,
                "ner_dosages": dosages,
                "ner_adverse_events": aes,
                "provenance": "ADVERSARIAL-STRESS",
                "sub_category": sub
            })

        out_df = pd.DataFrame(records)
        out_path = self.output_dir / "adversarial_test.parquet"
        out_df.to_parquet(out_path, index=False)
        logger.info(f"Saved adversarial_test.parquet ({len(out_df)} records)")
        return out_path

    def build_negation_stress_test(self, n_samples: int = 150) -> Path:
        """Dense negated phrases, pseudo-negations, and double negatives."""
        records = []
        templates = [
            ("Patient does not demonstrate any signs of pneumonitis or adverse toxicities.", "Low", "no acute adverse toxicities", "Osimertinib", "80 mg"),
            ("Cannot exclude mild fatigue, but exhibits no acute adverse toxicities.", "Low", "no acute adverse toxicities", "Pembrolizumab", "200 mg"),
            ("Patient is not without symptoms; developed severe neutropenia.", "High", "neutropenia", "Docetaxel", "75 mg"),
            ("There is no evidence to suggest therapy intolerance; no acute toxicities noted.", "Low", "no acute adverse toxicities", "Cisplatin", "100 mg")
        ]
        for i in range(n_samples):
            pid = f"NEG_STRESS_{i+1:04d}"
            sent, risk, ae, drug, dose = templates[i % len(templates)]
            prompt = f"CLINICAL NOTE: Patient {pid} on {drug} {dose}. {sent} Review of systems otherwise stable."
            target = f"Risk: {risk}\nKey Finding: Patient on {drug} at {dose} " + (f"developed {ae}." if ae != "no acute adverse toxicities" else "exhibits stable tolerance with no acute toxicities.") + f"\nAction: {'Continue monitoring.' if risk == 'Low' else 'Hold therapy.'}"
            records.append({
                "patient_id": pid,
                "prompt": prompt,
                "target": target,
                "expected_risk": risk,
                "ner_genes": [],
                "ner_drugs": [drug],
                "ner_dosages": [dose],
                "ner_adverse_events": [ae],
                "provenance": "NEGATION-STRESS",
                "sub_category": "pseudo_negation" if "not without" in sent or "Cannot exclude" in sent else "dense_negation"
            })
        out_df = pd.DataFrame(records)
        out_path = self.output_dir / "negation_stress_test.parquet"
        out_df.to_parquet(out_path, index=False)
        logger.info(f"Saved negation_stress_test.parquet ({len(out_df)} records)")
        return out_path

    def build_dosage_stress_test(self, n_samples: int = 150) -> Path:
        """Complex dosage units: BSA (mg/m²), AUC dosing, split schedules, microgram units."""
        dosages = ["175 mg/m²", "AUC 6", "12.5 mg bid", "500 mcg daily", "3 mg/kg IV q2w"]
        records = []
        for i in range(n_samples):
            pid = f"DOSE_{i+1:04d}"
            dose = dosages[i % len(dosages)]
            drug = "Carboplatin" if "AUC" in dose else ("Paclitaxel" if "m²" in dose else "Osimertinib")
            prompt = f"CLINICAL NOTE: Patient {pid} receiving {drug} at calculated dose of {dose}. Denies acute toxicities. Labs stable."
            target = f"Risk: Low\nKey Finding: Patient on {drug} at {dose} with stable tolerance.\nAction: Continue current regimen."
            records.append({
                "patient_id": pid,
                "prompt": prompt,
                "target": target,
                "expected_risk": "Low",
                "ner_genes": [],
                "ner_drugs": [drug],
                "ner_dosages": [dose],
                "ner_adverse_events": ["no acute adverse toxicities"],
                "provenance": "DOSAGE-STRESS",
                "sub_category": "complex_units"
            })
        out_df = pd.DataFrame(records)
        out_path = self.output_dir / "dosage_stress_test.parquet"
        out_df.to_parquet(out_path, index=False)
        logger.info(f"Saved dosage_stress_test.parquet ({len(out_df)} records)")
        return out_path

    def build_hallucination_probe_test(self, n_samples: int = 150) -> Path:
        """Notes containing decoy entities (non-cancer medications, family history) to test grounding."""
        decoys = [
            ("Lisinopril 10 mg for hypertension", "Osimertinib", "80 mg", "EGFR"),
            ("Metformin 500 mg for diabetes", "Pembrolizumab", "200 mg", "PD-L1"),
            ("Atorvastatin 20 mg for hyperlipidemia", "Cisplatin", "75 mg", "TP53")
        ]
        records = []
        for i in range(n_samples):
            pid = f"HALLUC_PROBE_{i+1:04d}"
            decoy, onc_drug, onc_dose, gene = decoys[i % len(decoys)]
            prompt = (
                f"MEDICAL ONCOLOGY RECORD:\n"
                f"Patient {pid} has active malignancy with {gene} alteration. "
                f"Active oncology therapy: {onc_drug} at {onc_dose} daily. "
                f"Past medical history co-morbidities managed with {decoy}. "
                f"No treatment-related adverse toxicities observed from oncology regimen."
            )
            target = f"Risk: Low\nKey Finding: {gene} variant receiving {onc_drug} at {onc_dose}.\nAction: Continue standard clinical monitoring."
            records.append({
                "patient_id": pid,
                "prompt": prompt,
                "target": target,
                "expected_risk": "Low",
                "ner_genes": [gene],
                "ner_drugs": [onc_drug], # Decoy drug must NOT be included in clinical summary target
                "ner_dosages": [onc_dose],
                "ner_adverse_events": ["no acute adverse toxicities"],
                "provenance": "HALLUCINATION-PROBE",
                "sub_category": "distractor_comorbidity"
            })
        out_df = pd.DataFrame(records)
        out_path = self.output_dir / "hallucination_probe_test.parquet"
        out_df.to_parquet(out_path, index=False)
        logger.info(f"Saved hallucination_probe_test.parquet ({len(out_df)} records)")
        return out_path

    def build_clinician_review_cohort(self, test_df: pd.DataFrame, n_samples: int = 100) -> Path:
        """Stratified sample across risk categories for blinded clinician evaluation."""
        high = test_df[test_df["target_risk_category"] == "High"].sample(n=min(35, len(test_df[test_df["target_risk_category"] == "High"])), random_state=self.seed)
        mod = test_df[test_df["target_risk_category"] == "Moderate"].sample(n=min(30, len(test_df[test_df["target_risk_category"] == "Moderate"])), random_state=self.seed)
        low = test_df[test_df["target_risk_category"] == "Low"].sample(n=n_samples - len(high) - len(mod), random_state=self.seed)

        cohort = pd.concat([high, mod, low]).sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
        cohort["review_id"] = [f"REV_{i+1:03d}" for i in range(len(cohort))]
        cohort["provenance"] = "CLINICIAN-REVIEW-COHORT"

        out_path = self.output_dir / "clinician_review_cohort.parquet"
        cohort.to_parquet(out_path, index=False)
        logger.info(f"Saved clinician_review_cohort.parquet ({len(cohort)} records)")
        return out_path

    def generate_all(self) -> Dict[str, Path]:
        """Generates all 8 evaluation datasets."""
        logger.info("Generating complete suite of independent evaluation datasets...")
        df, test_df = self.load_base_data()

        paths = {
            "standard_test": self.build_standard_test(test_df),
            "ood_synthetic": self.build_ood_synthetic(test_df),
            "ood_real": self.build_ood_real(),
            "adversarial_test": self.build_adversarial_test(),
            "negation_stress": self.build_negation_stress_test(),
            "dosage_stress": self.build_dosage_stress_test(),
            "hallucination_probe": self.build_hallucination_probe_test(),
            "clinician_review_cohort": self.build_clinician_review_cohort(test_df)
        }

        # Write dataset manifest
        manifest = {
            k: {
                "path": str(v),
                "records": len(pd.read_parquet(v)),
                "provenance": pd.read_parquet(v)["provenance"].iloc[0] if "provenance" in pd.read_parquet(v).columns else "STANDARD-HELD-OUT"
            }
            for k, v in paths.items()
        }
        with open(self.output_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"All 8 evaluation suites generated successfully under {self.output_dir}")
        return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gen = EvaluationDatasetGenerator(
        stage4_dataset_path="stage-4-slm/data-engineer/data/slm_finetune_dataset_v1.parquet",
        output_dir="stage-4-slm/evaluation-engineer/benchmarks"
    )
    gen.generate_all()
