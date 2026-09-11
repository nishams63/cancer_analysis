"""Tests for Retrieval Query Library Generation and YAML Export."""
import os
import yaml
import pytest
from src.prompts.retrieval_query_builder import (
    build_retrieval_query_library, export_retrieval_query_library_yaml
)
from src.prompts.scenario_builder import get_core_scenarios


def test_retrieval_query_library_structure():
    lib = build_retrieval_query_library()
    assert "version" in lib
    assert "target_knowledge_bases" in lib
    assert "intents" in lib
    assert len(lib["intents"]) >= 15


def test_each_intent_has_required_fields():
    lib = build_retrieval_query_library()
    for intent in lib["intents"]:
        assert intent["scenario_id"]
        assert intent["primary_query"]
        assert len(intent["search_terms"]) > 0
        assert intent["min_documents"] >= 1
        assert "target_domain" in intent


def test_yaml_export(tmp_path):
    out_file = str(tmp_path / "test_retrieval.yaml")
    res = export_retrieval_query_library_yaml(out_file)
    assert os.path.exists(res)
    with open(res, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert len(data["intents"]) >= 15