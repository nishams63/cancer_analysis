"""Cross-stage disagreement and discordance pattern analysis."""
import pandas as pd
from typing import List, Dict, Any

class DisagreementAnalyzer:
    def analyze(self) -> pd.DataFrame:
        disagreements = [
            {
                "pattern_id": "DIS-001",
                "input_category": "Tabular Low vs Clinical Text Acute Toxicity",
                "stage1_output": "Low Toxicity Risk (Confidence 0.62)",
                "stage2_output": "Stable Tumor Trajectory (0.78)",
                "stage3_output": "CRITICAL Urgency / HEMATOLOGIC Hazard (0.91)",
                "stage4_output": "Urgent hospitalization recommended for Febrile Neutropenia",
                "disagreement_type": "cross_modal_acuity_divergence",
                "confidence_spread": 0.29,
                "candidate_scenario": "PROMPT-R12",
                "root_cause": "Routine lab panel drawn 4 days prior missed acute weekend onset of febrile neutropenia."
            },
            {
                "pattern_id": "DIS-002",
                "input_category": "Tabular Nephrotoxicity Alert vs SLM Treatment Continuance",
                "stage1_output": "High Toxicity Risk (Creatinine 3.8 mg/dL, Confidence 0.84)",
                "stage2_output": "N/A",
                "stage3_output": "HIGH Urgency / RENAL Hazard (0.82)",
                "stage4_output": "Continue Cisplatin chemotherapy without dose modification",
                "disagreement_type": "safety_contraindication_incoherence",
                "confidence_spread": 0.35,
                "candidate_scenario": "PROMPT-R08",
                "root_cause": "SLM fails to synthesize laboratory contraindications with antineoplastic pharmacology."
            },
            {
                "pattern_id": "DIS-003",
                "input_category": "Histopathology Progression vs Flat Biomarker Trajectory",
                "stage1_output": "Low Toxicity Risk (0.75)",
                "stage2_output": "Visual Margin Progression (Attention 0.88) vs Flat ctDNA (0.52)",
                "stage3_output": "LOW Urgency / NONE Hazard (0.85)",
                "stage4_output": "Stable disease under ongoing oral maintenance therapy",
                "disagreement_type": "spatial_vs_serological_progression_conflict",
                "confidence_spread": 0.36,
                "candidate_scenario": "PROMPT-R04",
                "root_cause": "Discordant biological signals: cellular atypia at resection margin without systemic ctDNA shedding."
            }
        ]
        return pd.DataFrame(disagreements)
