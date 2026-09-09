"""
Unit tests for Upgraded Clinical Feature Pipeline and Whitespace Invariance.
"""

import pytest
import numpy as np
import pandas as pd
from token_alignment import canonicalize_text
from upgraded_feature_pipeline import (
    extract_upgraded_concept_counts,
    UpgradedClinicalFeaturePipeline,
)


def test_whitespace_invariance_char_count():
    single_spaced = "Patient reports grade 3 peripheral neuropathy and fatigue."
    double_spaced = "Patient  reports  grade 3  peripheral  neuropathy  and  fatigue.  "
    tabbed = "Patient\treports\tgrade 3\tperipheral\tneuropathy\tand\tfatigue.\n\n"
    
    # Without canonicalization, lengths differ significantly
    assert len(single_spaced) != len(double_spaced)
    assert len(single_spaced) != len(tabbed)
    
    # With canonicalization, lengths and content are strictly identical
    c_single = canonicalize_text(single_spaced)
    c_double = canonicalize_text(double_spaced)
    c_tabbed = canonicalize_text(tabbed)
    
    assert c_single == c_double
    assert c_single == c_tabbed
    assert len(c_single) == len(c_double) == len(c_tabbed)


def test_concept_counts_schema():
    class DummyConceptModel:
        def predict_entities(self, text, assign_polarity=True):
            return [
                {"label": "ADVERSE_EVENT", "polarity": "AFFIRMED"},
                {"label": "ADVERSE_EVENT", "polarity": "NEGATED"},
                {"label": "DRUG_NAME", "polarity": "AFFIRMED"},
                {"label": "DOSAGE", "polarity": "AFFIRMED"},
                {"label": "GENE_MUTATION", "polarity": "AFFIRMED"},
            ]

    df = pd.DataFrame([{"text": "Patient text here"}])
    counts_df = extract_upgraded_concept_counts(df, DummyConceptModel())
    
    expected_cols = [
        "document_id",
        "word_count",
        "char_count",
        "total_concepts",
        "affirmed_concepts",
        "negated_concepts",
        "historical_concepts",
        "drug_mentions",
        "mutation_mentions",
        "dosage_mentions",
        "adverse_event_mentions",
        "has_grade_3_4",
        "has_critical_symptom",
    ]
    assert list(counts_df.columns) == expected_cols
    assert counts_df.loc[0, "affirmed_concepts"] == 4
    assert counts_df.loc[0, "negated_concepts"] == 1
    assert counts_df.loc[0, "total_concepts"] == 5
    assert counts_df.loc[0, "drug_mentions"] == 1
    assert counts_df.loc[0, "dosage_mentions"] == 1
    assert counts_df.loc[0, "mutation_mentions"] == 1
    assert counts_df.loc[0, "adverse_event_mentions"] == 2


def test_upgraded_feature_pipeline_dimensions():
    train_texts = [
        "Patient has metastatic adenocarcinoma with EGFR mutation treated with osimertinib 80mg daily.",
        "Experiencing grade 2 rash and diarrhea, denying nausea.",
        "Follow-up after radiation therapy, stable disease on imaging.",
    ]
    df_train = pd.DataFrame({"text": train_texts, "document_id": ["d1", "d2", "d3"]})
    
    class DummyConceptModel:
        def predict_entities(self, text, assign_polarity=True):
            return []
        def extract_contextual_embeddings(self, texts, batch_size=32):
            return np.random.randn(len(texts), 384).astype(np.float32)

    model = DummyConceptModel()
    pipe = UpgradedClinicalFeaturePipeline(max_tfidf_features=100)
    struct_train_df = extract_upgraded_concept_counts(df_train, model)
    pipe.fit(df_train, struct_train_df)
    
    # Check dimensions for Experiments B, C, D
    feat_b = pipe.transform_experiment(df_train, struct_train_df, model, experiment_type="B")
    assert feat_b.shape == (3, 12)
    
    feat_c = pipe.transform_experiment(df_train, struct_train_df, model, experiment_type="C")
    assert feat_c.shape == (3, 384 + 12)
    
    feat_d = pipe.transform_experiment(df_train, struct_train_df, model, experiment_type="D")
    vocab_len = len(pipe.tfidf.vocabulary_)
    assert feat_d.shape == (3, vocab_len + 384 + 12)

