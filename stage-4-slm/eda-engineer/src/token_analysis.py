"""
Token-Length and Critical-Entity Truncation Analysis Module for Stage 4 EDA.
Computes BPE subword token distributions, context utilization, overflow rates,
and audits critical medical entity placement relative to context boundaries.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import tiktoken

logger = logging.getLogger("stage4_eda.token_analysis")


class TokenAnalyzer:
    """Performs BPE token profiling and context overflow analysis."""

    def __init__(self, tokenizer_model: str = "cl100k_base", context_limits: Optional[List[int]] = None):
        self.tokenizer_model = tokenizer_model
        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer_model)
        except Exception as e:
            logger.warning(f"Could not load tiktoken encoding '{tokenizer_model}': {e}. Falling back to cl100k_base.")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.context_limits = context_limits or [512, 1024, 2048, 4096]

    def count_tokens(self, text: str) -> int:
        """Counts tokens for a string using the BPE tokenizer."""
        if not text or not isinstance(text, str):
            return 0
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def analyze_token_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates descriptive statistics for source, target, and full prompt tokens.
        """
        # Vectorized or fast mapped token counting
        source_tokens = df["clinical_note"].fillna("").apply(self.count_tokens).values
        instr_tokens = df["instruction"].fillna("").apply(self.count_tokens).values
        
        target_risk_tokens = df["target_risk"].fillna("").apply(self.count_tokens).values
        target_kf_tokens = df["target_key_finding"].fillna("").apply(self.count_tokens).values
        target_act_tokens = df["target_action"].fillna("").apply(self.count_tokens).values
        target_total_tokens = target_risk_tokens + target_kf_tokens + target_act_tokens

        full_seq_tokens = source_tokens + instr_tokens + target_total_tokens

        stats = {
            "source_tokens": self._compute_summary_stats(source_tokens),
            "instruction_tokens": self._compute_summary_stats(instr_tokens),
            "target_risk_tokens": self._compute_summary_stats(target_risk_tokens),
            "target_key_finding_tokens": self._compute_summary_stats(target_kf_tokens),
            "target_action_tokens": self._compute_summary_stats(target_act_tokens),
            "target_total_tokens": self._compute_summary_stats(target_total_tokens),
            "full_sequence_tokens": self._compute_summary_stats(full_seq_tokens),
        }

        # Context limit analysis
        overflow_analysis = {}
        total_records = len(df)
        for limit in self.context_limits:
            over_count = int(np.sum(full_seq_tokens > limit))
            over_pct = float(round((over_count / total_records) * 100.0, 4)) if total_records > 0 else 0.0
            utilization_pct = float(round(np.mean(full_seq_tokens / limit) * 100.0, 2)) if total_records > 0 else 0.0
            overflow_analysis[f"limit_{limit}"] = {
                "context_limit": limit,
                "records_over_context_limit": over_count,
                "overflow_percentage": over_pct,
                "context_utilization_percentage": utilization_pct
            }

        stats["context_limits_evaluation"] = overflow_analysis

        # Add token length series back to a lightweight dict for visualization
        token_arrays = {
            "source": source_tokens,
            "target": target_total_tokens,
            "full_sequence": full_seq_tokens
        }

        return {"metrics": stats, "token_arrays": token_arrays}

    @staticmethod
    def _compute_summary_stats(arr: np.ndarray) -> Dict[str, float]:
        """Calculates min, max, mean, median, std, P90, P95, P99 for an array of integers."""
        if len(arr) == 0:
            return {"min": 0, "max": 0, "mean": 0, "median": 0, "std": 0, "p90": 0, "p95": 0, "p99": 0}
        return {
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
            "mean": float(round(float(np.mean(arr)), 2)),
            "median": float(round(float(np.median(arr)), 2)),
            "std": float(round(float(np.std(arr)), 2)),
            "p90": float(round(float(np.percentile(arr, 90)), 2)),
            "p95": float(round(float(np.percentile(arr, 95)), 2)),
            "p99": float(round(float(np.percentile(arr, 99)), 2)),
        }

    def analyze_critical_entity_truncation_risk(
        self, df: pd.DataFrame, target_context_limit: int = 4096, tail_fraction: float = 0.10
    ) -> Dict[str, Any]:
        """
        Audits whether critical clinical entities (Gene, Drug, Dosage, Adverse Event)
        occur in the tail boundary of notes that approach or exceed the context limit.
        """
        potential_truncation_count = 0
        boundary_records = []
        total_records = len(df)

        for idx, row in df.iterrows():
            note = str(row.get("clinical_note", ""))
            total_tokens = self.count_tokens(note)

            # If note tokens exceed or fall within the tail threshold of target limit
            threshold = target_context_limit * (1.0 - tail_fraction)
            if total_tokens >= threshold:
                # Check if entities occur in the last tail_fraction of the text
                note_len = len(note)
                tail_start_char = int(note_len * (1.0 - tail_fraction))
                tail_text = note[tail_start_char:].lower()

                # Gather entities
                all_entities = []
                for col in ["ner_genes", "ner_drugs", "ner_dosages", "ner_adverse_events"]:
                    ents = row.get(col, [])
                    if isinstance(ents, (list, np.ndarray)):
                        all_entities.extend([str(e).lower() for e in ents if str(e).lower() not in ("none/unknown", "none", "")])

                tail_entities = [e for e in set(all_entities) if e in tail_text]
                if tail_entities:
                    potential_truncation_count += 1
                    boundary_records.append({
                        "note_id": row.get("note_id", f"row_{idx}"),
                        "total_tokens": total_tokens,
                        "tail_entities": tail_entities
                    })

        truncation_rate = float(round((potential_truncation_count / total_records) * 100.0, 4)) if total_records > 0 else 0.0

        return {
            "target_context_limit": target_context_limit,
            "tail_fraction": tail_fraction,
            "potential_entity_truncation_count": potential_truncation_count,
            "potential_entity_truncation_rate": truncation_rate,
            "flagged_boundary_sample_count": len(boundary_records)
        }
