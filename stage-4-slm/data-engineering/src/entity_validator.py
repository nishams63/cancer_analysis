"""
Entity Normalization and Preservation Quality Gate Module for Stage 4.
Implements:
- Section 4a Normalization Rules (case, whitespace, unicode NFKC, dosage format/units/range, drug synonyms)
- Section 6 Entity-Preservation Quality Gate (Missing, Invented, Correct Preservation)
- Section 6a NER Confidence Awareness and Calibration
"""

import re
import yaml
import logging
import unicodedata
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional
import pandas as pd

logger = logging.getLogger("stage4.entity_validator")


class EntityNormalizer:
    """Handles clinical normalization rules safely without changing medical meaning."""

    def __init__(self, drug_synonyms_path: Optional[str] = None):
        self.drug_synonyms = self._load_drug_synonyms(drug_synonyms_path)
        self.normalization_logs: List[Dict[str, Any]] = []

    def _load_drug_synonyms(self, path: Optional[str]) -> Dict[str, Set[str]]:
        synonym_map: Dict[str, Set[str]] = {}
        if not path or not Path(path).exists():
            return synonym_map

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        syn_dict = data.get("synonyms", {})
        for generic, brand_list in syn_dict.items():
            canonical = generic.strip().lower()
            all_aliases = {canonical}
            for b in brand_list:
                all_aliases.add(b.strip().lower())
            
            # Map each alias to the full set
            for alias in all_aliases:
                synonym_map[alias] = all_aliases

        return synonym_map

    @staticmethod
    def clean_unicode_and_case(text: str) -> str:
        """Applies NFKC unicode normalization, whitespace stripping, and lowercasing."""
        if not text:
            return ""
        norm = unicodedata.normalize("NFKC", str(text))
        norm = re.sub(r"\s+", " ", norm).strip().lower()
        return norm

    @staticmethod
    def parse_dosage(dosage_str: str) -> Optional[Dict[str, Any]]:
        """
        Parses dosage string into numeric values, range (if any), and unit.
        Examples:
          '5 mg' -> {'min': 5.0, 'max': 5.0, 'unit': 'mg', 'is_range': False}
          '5-10 mg' / '5–10 mg' -> {'min': 5.0, 'max': 10.0, 'unit': 'mg', 'is_range': True}
          '5000 mcg' -> {'min': 5000.0, 'max': 5000.0, 'unit': 'mcg', 'is_range': False}
        """
        if not dosage_str:
            return None

        clean = unicodedata.normalize("NFKC", str(dosage_str)).strip().lower()
        # Handle en-dash and em-dash
        clean = clean.replace("–", "-").replace("—", "-")

        # Match range pattern: e.g. 5-10 mg or 5 - 10 mg/dL
        range_match = re.match(r"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*([a-z/%]+(?:/[a-z0-9]+)?)$", clean)
        if range_match:
            min_val = float(range_match.group(1))
            max_val = float(range_match.group(2))
            unit = range_match.group(3).strip()
            return {"min": min_val, "max": max_val, "unit": unit, "is_range": True}

        # Match single dosage: e.g. 5 mg, 265.4mg, 1.2 mg/dl
        single_match = re.match(r"^(\d+(?:\.\d+)?)\s*([a-z/%]+(?:/[a-z0-9]+)?)$", clean)
        if single_match:
            val = float(single_match.group(1))
            unit = single_match.group(2).strip()
            return {"min": val, "max": val, "unit": unit, "is_range": False}

        return None

    def match_dosage(self, source_dosage: str, target_text: str) -> bool:
        """
        Checks if source_dosage matches any dosage in target_text, handling:
        - spacing: '5 mg' vs '5mg'
        - range: '5-10 mg' matched by single value 7 mg
        - unit conversion: '5000 mcg' matched by '5 mg'
        """
        src_parsed = self.parse_dosage(source_dosage)
        target_norm = self.clean_unicode_and_case(target_text)

        # 1. Exact or whitespace-insensitive literal check
        src_clean = self.clean_unicode_and_case(source_dosage)
        src_no_space = src_clean.replace(" ", "")
        target_no_space = target_norm.replace(" ", "")
        if src_no_space in target_no_space:
            return True

        if not src_parsed:
            return False

        # Find all candidate dosages in target text
        # e.g., numbers followed by units
        candidates = re.findall(r"\b\d+(?:\.\d+)?\s*(?:mg/m2|mg/dl|mg|mcg|g/dl|g|mmhg|%)\b", target_norm)
        for cand in candidates:
            cand_parsed = self.parse_dosage(cand)
            if not cand_parsed:
                continue

            # Check unit equivalence
            u_src = src_parsed["unit"]
            u_cand = cand_parsed["unit"]

            # Same unit
            if u_src == u_cand:
                if src_parsed["is_range"]:
                    # Range match: candidate single value within source range
                    if src_parsed["min"] <= cand_parsed["min"] <= src_parsed["max"]:
                        return True
                elif cand_parsed["is_range"]:
                    if cand_parsed["min"] <= src_parsed["min"] <= cand_parsed["max"]:
                        return True
                else:
                    if abs(src_parsed["min"] - cand_parsed["min"]) < 1e-4:
                        return True

            # Safe unit conversion: mcg <-> mg
            if (u_src == "mcg" and u_cand == "mg") or (u_src == "mg" and u_cand == "mcg"):
                val_src_mg = src_parsed["min"] / 1000.0 if u_src == "mcg" else src_parsed["min"]
                val_cand_mg = cand_parsed["min"] / 1000.0 if u_cand == "mcg" else cand_parsed["min"]
                if abs(val_src_mg - val_cand_mg) < 1e-4:
                    self.normalization_logs.append({
                        "rule": "unit_conversion_mcg_mg",
                        "source": source_dosage,
                        "target_match": cand,
                        "status": "normalized"
                    })
                    return True

            # Safe unit conversion: mg <-> g
            if (u_src == "mg" and u_cand == "g") or (u_src == "g" and u_cand == "mg"):
                val_src_g = src_parsed["min"] / 1000.0 if u_src == "mg" else src_parsed["min"]
                val_cand_g = cand_parsed["min"] / 1000.0 if u_cand == "mg" else cand_parsed["min"]
                if abs(val_src_g - val_cand_g) < 1e-4:
                    self.normalization_logs.append({
                        "rule": "unit_conversion_mg_g",
                        "source": source_dosage,
                        "target_match": cand,
                        "status": "normalized"
                    })
                    return True

        return False

    def match_drug(self, drug_name: str, target_text: str) -> bool:
        """
        Checks drug presence in target_text using case normalization and
        explicit synonym lookup table. Anything not in table is treated as genuine mismatch.
        """
        target_norm = self.clean_unicode_and_case(target_text)
        drug_clean = self.clean_unicode_and_case(drug_name)

        if not drug_clean:
            return True

        # Check direct match with word boundaries
        if re.search(r"\b" + re.escape(drug_clean) + r"\b", target_norm):
            return True

        # Check versioned synonym table
        synonyms = self.drug_synonyms.get(drug_clean, set())
        for syn in synonyms:
            if re.search(r"\b" + re.escape(syn) + r"\b", target_norm):
                self.normalization_logs.append({
                    "rule": "drug_synonym_match",
                    "source_drug": drug_name,
                    "matched_synonym": syn
                })
                return True

        return False

    def match_gene(self, gene_name: str, target_text: str) -> bool:
        """Checks gene/mutation match with case and prefix tolerance."""
        target_norm = self.clean_unicode_and_case(target_text)
        gene_clean = self.clean_unicode_and_case(gene_name)

        if not gene_clean or gene_clean in ("none/unknown", "none", "unknown"):
            return True

        # Word boundary match
        pattern = r"\b" + re.escape(gene_clean) + r"\b"
        return bool(re.search(pattern, target_norm))

    def match_adverse_event(self, ae_name: str, target_text: str) -> bool:
        """Checks adverse event presence with sub-phrase matching."""
        target_norm = self.clean_unicode_and_case(target_text)
        ae_clean = self.clean_unicode_and_case(ae_name)

        if not ae_clean:
            return True

        # Direct containment
        if ae_clean in target_norm:
            return True

        # Token overlap for compound expressions (e.g. 'acute nephrotoxicity' in 'nephrotoxicity')
        ae_words = [w for w in ae_clean.split() if len(w) > 3 and w not in ("acute", "grade", "mild", "severe", "moderate")]
        if ae_words and all(w in target_norm for w in ae_words):
            return True

        return False


class EntityPreservationQualityGate:
    """
    Evaluates entity preservation across Gene, Drug, Dosage, Adverse Event.
    Detects missing and invented entities. Incorporates Stage 3 NER baseline F1 calibration.
    """

    # Stage 3 NER benchmark reference metrics (from Stage 3 Baseline Report)
    STAGE3_NER_PRECISION = 0.7191
    STAGE3_NER_RECALL = 0.8797
    STAGE3_NER_F1 = 0.7670

    def __init__(self, normalizer: Optional[EntityNormalizer] = None, drug_synonyms_path: Optional[str] = None):
        self.normalizer = normalizer or EntityNormalizer(drug_synonyms_path)

    def validate_example(
        self,
        source_note: str,
        target_risk: str,
        target_key_finding: str,
        target_action: str,
        ner_genes: List[str],
        ner_drugs: List[str],
        ner_dosages: List[str],
        ner_adverse_events: List[str]
    ) -> Dict[str, Any]:
        """
        Validates entity preservation between Stage 3 NER and generated target.
        Returns detailed audit dictionary.
        """
        combined_target = f"{target_risk} {target_key_finding} {target_action}"

        missing_genes = []
        for g in ner_genes:
            if not self.normalizer.match_gene(g, combined_target):
                missing_genes.append(g)

        missing_drugs = []
        for d in ner_drugs:
            if not self.normalizer.match_drug(d, combined_target):
                missing_drugs.append(d)

        missing_dosages = []
        for dose in ner_dosages:
            if not self.normalizer.match_dosage(dose, combined_target):
                missing_dosages.append(dose)

        missing_aes = []
        for ae in ner_adverse_events:
            if not self.normalizer.match_adverse_event(ae, combined_target):
                missing_aes.append(ae)

        total_entities = len(ner_genes) + len(ner_drugs) + len(ner_dosages) + len(ner_adverse_events)
        total_missing = len(missing_genes) + len(missing_drugs) + len(missing_dosages) + len(missing_aes)
        preserved_entities = total_entities - total_missing

        coverage = round(preserved_entities / total_entities, 4) if total_entities > 0 else 1.0

        # Check for Invented Entities (Entities in Target unsupported by Stage 3 NER or Source Note)
        invented_entities = self._detect_invented_entities(
            target_text=combined_target,
            source_note=source_note,
            ner_drugs=ner_drugs,
            ner_genes=ner_genes,
            ner_aes=ner_adverse_events
        )

        reasons = []
        if total_missing > 0:
            reasons.append(f"Missing entities ({total_missing}/{total_entities}): genes={missing_genes}, drugs={missing_drugs}, dosages={missing_dosages}, aes={missing_aes}")
        if invented_entities:
            reasons.append(f"Invented/unsupported entities detected: {invented_entities}")

        # Classification decision: PASS / REJECT / FLAG
        if total_missing == 0 and not invented_entities:
            status = "PASS"
            validation_reason = "All Stage 3 entities preserved without hallucinations"
        elif total_missing > 0 or invented_entities:
            # If minor boundary mismatch on secondary AE but primary drug & gene intact, can FLAG; else REJECT
            if len(missing_drugs) == 0 and len(missing_genes) == 0 and not invented_entities and coverage >= 0.75:
                status = "FLAG"
                validation_reason = "; ".join(reasons)
            else:
                status = "REJECT"
                validation_reason = "; ".join(reasons)

        return {
            "entity_check_status": status,
            "entity_coverage": coverage,
            "missing_genes": missing_genes,
            "missing_drugs": missing_drugs,
            "missing_dosages": missing_dosages,
            "missing_adverse_events": missing_aes,
            "invented_entities": invented_entities,
            "validation_reason": validation_reason,
            "stage3_ner_f1_baseline": self.STAGE3_NER_F1
        }

    def _detect_invented_entities(
        self,
        target_text: str,
        source_note: str,
        ner_drugs: List[str],
        ner_genes: List[str],
        ner_aes: List[str]
    ) -> List[str]:
        """
        Scans generated target for known medical entities that are NOT present
        in the source note or Stage 3 NER entities.
        """
        invented = []
        target_norm = self.normalizer.clean_unicode_and_case(target_text)
        source_norm = self.normalizer.clean_unicode_and_case(source_note)

        # Known candidate oncology drug list
        known_drugs = [
            "cisplatin", "carboplatin", "oxaliplatin", "paclitaxel", "docetaxel",
            "pembrolizumab", "nivolumab", "atezolizumab", "durvalumab",
            "osimertinib", "erlotinib", "gefitinib", "alectinib", "crizotinib",
            "fluorouracil", "5-fu", "capecitabine", "gemcitabine", "doxorubicin",
            "trastuzumab", "pertuzumab", "tamoxifen", "letrozole", "enzalutamide",
            "abiraterone", "irinotecan", "etoposide", "warfarin", "coumadin"
        ]

        for drug in known_drugs:
            # If drug is in target
            if re.search(r"\b" + re.escape(drug) + r"\b", target_norm):
                # Must be supported by source note OR ner_drugs (accounting for synonyms)
                is_in_source = bool(re.search(r"\b" + re.escape(drug) + r"\b", source_norm))
                is_in_ner = any(self.normalizer.match_drug(drug, nd) or self.normalizer.match_drug(nd, drug) for nd in ner_drugs)
                if not is_in_source and not is_in_ner:
                    invented.append(f"drug:{drug}")

        # Known driver genes
        known_genes = [
            "egfr", "kras", "tp53", "braf", "alk", "pik3ca", "her2", "erbb2",
            "brca1", "brca2", "ros1", "met", "ret", "cyp2c9"
        ]
        for gene in known_genes:
            if re.search(r"\b" + re.escape(gene) + r"\b", target_norm):
                is_in_source = bool(re.search(r"\b" + re.escape(gene) + r"\b", source_norm))
                is_in_ner = any(gene in self.normalizer.clean_unicode_and_case(ng) for ng in ner_genes)
                if not is_in_source and not is_in_ner:
                    invented.append(f"gene:{gene}")

        return invented

    def evaluate_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Validates entity preservation across entire dataset.
        Returns:
            pass_df: Records passing quality gate
            reject_df: Records rejected or flagged
            metrics: Aggregate coverage and validation statistics
        """
        logger.info("Evaluating entity-preservation quality gate for %d records...", len(df))
        results = []

        for idx, row in df.iterrows():
            source_note = str(row.get("text", row.get("clinical_note", "")))
            t_risk = str(row.get("target_risk", ""))
            t_kf = str(row.get("target_key_finding", ""))
            t_action = str(row.get("target_action", ""))
            genes = row.get("ner_genes", [])
            drugs = row.get("ner_drugs", [])
            dosages = row.get("ner_dosages", [])
            aes = row.get("ner_adverse_events", [])

            res = self.validate_example(
                source_note=source_note,
                target_risk=t_risk,
                target_key_finding=t_kf,
                target_action=t_action,
                ner_genes=genes,
                ner_drugs=drugs,
                ner_dosages=dosages,
                ner_adverse_events=aes
            )
            results.append(res)

        df_eval = pd.DataFrame(results)
        df_augmented = df.copy()

        for col in df_eval.columns:
            df_augmented[col] = df_eval[col].values

        pass_mask = df_augmented["entity_check_status"] == "PASS"
        pass_df = df_augmented[pass_mask].copy()
        reject_df = df_augmented[~pass_mask].copy()

        total = len(df_augmented)
        pass_count = int(pass_mask.sum())
        reject_count = total - pass_count
        rejection_rate = round(reject_count / total, 4) if total > 0 else 0.0

        metrics = {
            "total_evaluated": total,
            "pass_count": pass_count,
            "reject_count": reject_count,
            "rejection_rate": rejection_rate,
            "mean_entity_coverage": round(float(df_augmented["entity_coverage"].mean()), 4) if total > 0 else 0.0,
            "stage3_ner_f1_baseline": self.STAGE3_NER_F1,
            "stage3_ner_precision": self.STAGE3_NER_PRECISION,
            "stage3_ner_recall": self.STAGE3_NER_RECALL,
            "total_normalizations_logged": len(self.normalizer.normalization_logs)
        }

        logger.info(
            "Quality gate completed: PASS=%d, REJECT/FLAG=%d, RejectionRate=%.2f%%, MeanCoverage=%.2f%%",
            pass_count,
            reject_count,
            rejection_rate * 100,
            metrics["mean_entity_coverage"] * 100
        )

        return pass_df, reject_df, metrics
