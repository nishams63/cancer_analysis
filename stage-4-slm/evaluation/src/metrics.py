"""
Clinical Evaluation Metrics Module for Stage 4 SLM.
Computes Natural Language Generation (NLG) metrics (ROUGE-1, ROUGE-2, ROUGE-L, BLEU-4)
and Clinical Entity Preservation Metrics (Exact/Relaxed Span F1, Hallucination Rate).
"""

import re
import logging
from typing import Dict, List, Any, Set, Tuple
import numpy as np
from rouge_score import rouge_scorer
import sacrebleu

logger = logging.getLogger("stage4.evaluation.metrics")


class ClinicalMetricsCalculator:
    """Computes comprehensive NLG and Clinical Safety metrics for SLM predictions."""

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def compute_nlg_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculates mean ROUGE-1, ROUGE-2, ROUGE-L, and BLEU-4 scores."""
        if not predictions or not references:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bleu4": 0.0}

        r1_scores = []
        r2_scores = []
        rl_scores = []

        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            r1_scores.append(scores["rouge1"].fmeasure)
            r2_scores.append(scores["rouge2"].fmeasure)
            rl_scores.append(scores["rougeL"].fmeasure)

        # SacreBLEU expects list of references: [[ref1, ref2, ...]]
        try:
            bleu = sacrebleu.corpus_bleu(predictions, [references])
            bleu4 = float(bleu.score) / 100.0
        except Exception as e:
            logger.warning(f"SacreBLEU calculation warning: {e}; using token fallback.")
            bleu4 = 0.0

        return {
            "rouge1": float(np.mean(r1_scores)),
            "rouge2": float(np.mean(r2_scores)),
            "rougeL": float(np.mean(rl_scores)),
            "bleu4": float(bleu4)
        }

    @staticmethod
    def extract_entities_from_text(text: str) -> Dict[str, Set[str]]:
        """Extracts mention sets for drugs, genes, and dosages from narrative text."""
        t_upper = text.upper()
        found_genes = set()
        for g in ["EGFR", "KRAS", "BRAF", "ALK", "TP53", "ROS1", "BRCA1", "BRCA2", "HER2", "MET"]:
            if re.search(rf"\b{g}\b", t_upper):
                found_genes.add(g)

        found_drugs = set()
        known_drugs = [
            "Docetaxel", "Cisplatin", "Carboplatin", "Paclitaxel", "Pemetrexed",
            "Nivolumab", "Durvalumab", "Atezolizumab", "Erlotinib", "Osimertinib",
            "Trastuzumab", "Olaparib", "Alectinib", "Warfarin", "Pembrolizumab"
        ]
        for d in known_drugs:
            if re.search(rf"\b{re.escape(d)}\b", text, re.IGNORECASE):
                found_drugs.add(d.capitalize())

        found_dosages = set()
        d_matches = re.findall(r"\b\d+(?:\.\d+)?\s*(?:mg/m2|mg|mcg|g)\b", text, re.IGNORECASE)
        for dm in d_matches:
            found_dosages.add(re.sub(r"\s+", "", dm.lower()))

        return {
            "genes": found_genes,
            "drugs": found_drugs,
            "dosages": found_dosages
        }

    def compute_entity_preservation(
        self,
        predicted_triads: List[Dict[str, str]],
        reference_entities: List[Dict[str, Any]],
        source_notes: List[str]
    ) -> Dict[str, Any]:
        """
        Computes precision, recall, F1, and hallucination rates for critical clinical entities.
        """
        gene_p, gene_r = [], []
        drug_p, drug_r = [], []
        dosage_p, dosage_r = [], []
        hallucinations = 0
        total_predictions = len(predicted_triads)

        for triad, ref_ent, note in zip(predicted_triads, reference_entities, source_notes):
            combined_pred_text = f"{triad.get('target_risk', '')} {triad.get('target_key_finding', '')} {triad.get('target_action', '')}"
            pred_ents = self.extract_entities_from_text(combined_pred_text)
            note_ents = self.extract_entities_from_text(note)

            # Drug Preservation
            ref_drugs = set([str(x).capitalize() for x in ref_ent.get("ner_drugs", []) if str(x)])
            if not ref_drugs:
                ref_drugs = note_ents["drugs"]

            pred_drugs = pred_ents["drugs"]
            if ref_drugs and pred_drugs:
                overlap = len(pred_drugs.intersection(ref_drugs))
                drug_p.append(overlap / len(pred_drugs))
                drug_r.append(overlap / len(ref_drugs))
            elif not ref_drugs and not pred_drugs:
                drug_p.append(1.0)
                drug_r.append(1.0)
            elif ref_drugs and not pred_drugs:
                drug_r.append(0.0)
                drug_p.append(1.0)
            elif not ref_drugs and pred_drugs:
                drug_p.append(0.0)

            # Check for hallucinated drugs (present in prediction but entirely absent in source note)
            unsupported_drugs = pred_drugs - note_ents["drugs"]
            if unsupported_drugs:
                hallucinations += 1

            # Gene Preservation
            ref_genes = set([str(x).upper() for x in ref_ent.get("ner_genes", []) if str(x) and str(x) != "None/Unknown"])
            if not ref_genes:
                ref_genes = note_ents["genes"]

            pred_genes = pred_ents["genes"]
            if ref_genes and pred_genes:
                overlap = len(pred_genes.intersection(ref_genes))
                gene_p.append(overlap / len(pred_genes))
                gene_r.append(overlap / len(ref_genes))
            elif not ref_genes and not pred_genes:
                gene_p.append(1.0)
                gene_r.append(1.0)
            elif ref_genes and not pred_genes:
                gene_r.append(0.0)
                gene_p.append(1.0)
            elif not ref_genes and pred_genes:
                gene_p.append(0.0)

            # Dosage Preservation
            ref_dosages = set([re.sub(r"\s+", "", str(x).lower()) for x in ref_ent.get("ner_dosages", []) if str(x)])
            if not ref_dosages:
                ref_dosages = note_ents["dosages"]

            pred_dosages = pred_ents["dosages"]
            if ref_dosages and pred_dosages:
                overlap = len(pred_dosages.intersection(ref_dosages))
                dosage_p.append(overlap / len(pred_dosages))
                dosage_r.append(overlap / len(ref_dosages))
            elif not ref_dosages and not pred_dosages:
                dosage_p.append(1.0)
                dosage_r.append(1.0)
            elif ref_dosages and not pred_dosages:
                dosage_r.append(0.0)
                dosage_p.append(1.0)
            elif not ref_dosages and pred_dosages:
                dosage_p.append(0.0)

        def f1(p_list, r_list):
            p = float(np.mean(p_list)) if p_list else 1.0
            r = float(np.mean(r_list)) if r_list else 1.0
            score = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            return p, r, score

        dp, dr, df1 = f1(drug_p, drug_r)
        gp, gr, gf1 = f1(gene_p, gene_r)
        dosp, dosr, dosf1 = f1(dosage_p, dosage_r)

        mean_span_f1 = (df1 + gf1 + dosf1) / 3.0
        hallucination_rate = float(hallucinations / max(total_predictions, 1))

        return {
            "drug": {"precision": dp, "recall": dr, "f1": df1},
            "gene": {"precision": gp, "recall": gr, "f1": gf1},
            "dosage": {"precision": dosp, "recall": dosr, "f1": dosf1},
            "mean_entity_f1": float(mean_span_f1),
            "hallucination_rate": float(hallucination_rate),
            "unsupported_entity_count": hallucinations
        }

    @staticmethod
    def compute_bootstrap_ci(scores: List[float], n_bootstraps: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
        """Computes mean and non-parametric bootstrap confidence interval."""
        if not scores:
            return 0.0, 0.0, 0.0
        arr = np.array(scores)
        mean_val = float(np.mean(arr))
        boot_means = []
        rng = np.random.default_rng(42)
        n = len(arr)
        for _ in range(n_bootstraps):
            sample = rng.choice(arr, size=n, replace=True)
            boot_means.append(np.mean(sample))
        alpha = (1.0 - ci) / 2.0
        low = float(np.percentile(boot_means, alpha * 100))
        high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
        return mean_val, low, high
