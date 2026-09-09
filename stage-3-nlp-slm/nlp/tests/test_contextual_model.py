"""
Unit tests for ContextualClinicalConceptModel and clinical polarity assignment.
"""

import pytest
from contextual_concept_model import ContextualClinicalConceptModel
from negation_detection import resolve_concept_polarity


def test_contextual_model_init():
    model = ContextualClinicalConceptModel()
    assert model.model is not None
    assert model.tokenizer is not None
    assert len(model.model.config.id2label) == 9
    assert len(model.model.config.label2id) == 9


def test_freeze_lower_layers():
    model = ContextualClinicalConceptModel()
    model.freeze_lower_layers(num_layers_to_freeze=4)
    # Check that lower layers are frozen
    for i, layer in enumerate(model.model.bert.encoder.layer):
        for param in layer.parameters():
            if i < 4:
                assert not param.requires_grad
            else:
                assert param.requires_grad


def test_polarity_assignment_negated():
    text_negated = "Patient denies any nausea or vomiting during the current cycle."
    # "nausea or vomiting" is at index 19 to 37
    pol = resolve_concept_polarity(text_negated, 19, 37)
    assert pol == "NEGATED"


def test_polarity_assignment_affirmed():
    text_affirmed = "Patient reports persistent severe nausea after infusion."
    # "severe nausea" is at index 27 to 40
    pol = resolve_concept_polarity(text_affirmed, 27, 40)
    assert pol == "AFFIRMED"


def test_polarity_assignment_historical():
    text_hist = "Past history of severe peripheral neuropathy in 2021."
    # "severe peripheral neuropathy" is at index 16 to 44
    pol = resolve_concept_polarity(text_hist, 16, 44)
    assert pol == "HISTORICAL"
