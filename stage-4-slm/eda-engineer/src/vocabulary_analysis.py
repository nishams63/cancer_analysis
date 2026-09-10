"""
Vocabulary and Tokenizer Fragmentation Analysis Module for Stage 4 EDA.
Audits lexical diversity, hapax legomena, empirical clinical terms,
and measures BPE subword fragmentation across critical oncology entities.
"""

import re
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import pandas as pd
import tiktoken


class VocabularyAnalyzer:
    """Performs lexical profiling and tokenizer subword fragmentation audits."""

    def __init__(self, critical_terms_yaml: str, tokenizer_model: str = "cl100k_base"):
        self.critical_terms_path = Path(critical_terms_yaml)
        self.critical_terms = self._load_critical_terms()
        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer_model)
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _load_critical_terms(self) -> Dict[str, List[str]]:
        if not self.critical_terms_path.exists():
            return {"drugs": [], "biomarkers": [], "regimens": [], "toxicities": [], "clinical_acronyms": []}
        with open(self.critical_terms_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def profile_vocabulary(self, df: pd.DataFrame, max_frequent: int = 50) -> Dict[str, Any]:
        """
        Profiles empirical word frequencies, total word count, unique vocabulary,
        hapax legomena (words appearing once), and frequency of critical medical terms.
        """
        word_counter = Counter()
        for text in df["clinical_note"].dropna():
            words = re.findall(r"\b[A-Za-z0-9\+\-\/\.]+\b", text.lower())
            word_counter.update(words)

        total_words = sum(word_counter.values())
        unique_vocab_size = len(word_counter)
        hapax_words = [w for w, c in word_counter.items() if c == 1]
        hapax_count = len(hapax_words)
        hapax_rate = float(round((hapax_count / unique_vocab_size) * 100.0, 2)) if unique_vocab_size > 0 else 0.0

        top_frequent = word_counter.most_common(max_frequent)

        return {
            "total_words": total_words,
            "unique_vocabulary_size": unique_vocab_size,
            "hapax_legomena_count": hapax_count,
            "hapax_legomena_rate": hapax_rate,
            "top_frequent_words": [{"word": w, "count": c} for w, c in top_frequent]
        }

    def audit_tokenizer_fragmentation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Inspects subword tokenization behavior for critical clinical terms,
        computing token counts, subword token sequences, and fragmentation ratios.
        """
        term_audit = []
        full_corpus = " ".join(df["clinical_note"].dropna().tolist()).lower()

        flattened_terms = []
        for category, terms in self.critical_terms.items():
            for t in terms:
                flattened_terms.append((category, str(t).strip()))

        for category, term in flattened_terms:
            term_lower = term.lower()
            # Empirical corpus frequency (exact substring or regex boundary)
            pattern = rf"\b{re.escape(term_lower)}\b" if not term_lower.endswith("+") else rf"{re.escape(term_lower)}"
            corpus_freq = len(re.findall(pattern, full_corpus, flags=re.IGNORECASE))

            # Tokenizer encoding
            token_ids = self.tokenizer.encode(term, disallowed_special=())
            # Decode each token back to string
            token_strings = [self.tokenizer.decode([tid]) for tid in token_ids]
            token_count = len(token_ids)
            word_count = max(1, len(term.split()))
            frag_ratio = float(round(token_count / word_count, 2))

            is_fragmented = token_count >= 4 or frag_ratio >= 3.0

            term_audit.append({
                "term": term,
                "category": category,
                "frequency": corpus_freq,
                "token_count": token_count,
                "token_sequence": token_strings,
                "fragmentation_ratio": frag_ratio,
                "flagged_excessive_fragmentation": is_fragmented
            })

        flagged_count = sum(1 for t in term_audit if t["flagged_excessive_fragmentation"])

        return {
            "total_critical_terms_evaluated": len(term_audit),
            "flagged_excessive_fragmentation_count": flagged_count,
            "terms": term_audit
        }
