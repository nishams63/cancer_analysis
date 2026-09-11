"""Retrieval Query Library Builder: Generates structured RAG retrieval specifications."""
import os
from typing import List, Dict, Any
import yaml
from .scenario_builder import get_core_scenarios
from .prompt_schema import ScenarioDefinition


def build_retrieval_query_library(scenarios: List[ScenarioDefinition] = None) -> Dict[str, Any]:
    """Compile retrieval intents from scenarios into a comprehensive RAG query library."""
    if scenarios is None:
        scenarios = get_core_scenarios()

    query_library = {
        "version": "1.0.0",
        "description": "Evidence retrieval query library for oncology stress-test scenarios",
        "target_knowledge_bases": [
            "nccn_clinical_practice_guidelines",
            "fda_approved_drug_labels",
            "asco_emergency_triage_algorithms",
            "ajcc_cancer_staging_manual_8th",
            "ctcae_toxicity_criteria_v5"
        ],
        "intents": []
    }

    for sc in scenarios:
        rag = sc.rag_intent
        intent_entry = {
            "scenario_id": sc.scenario_id,
            "scenario_title": sc.title,
            "intent_id": rag.intent_id,
            "target_domain": rag.target_domain,
            "primary_query": rag.primary_query,
            "search_terms": rag.search_terms,
            "required_keywords": rag.required_keywords,
            "forbidden_keywords": rag.forbidden_keywords,
            "min_documents": rag.min_documents,
            "guideline_reference": rag.guideline_reference,
            "target_blind_spots": sc.target_blind_spots,
            "retrieval_scoring": {
                "must_match_all_required": True,
                "penalty_for_forbidden": 1.0,
                "min_semantic_similarity": 0.72
            }
        }
        query_library["intents"].append(intent_entry)

    return query_library


def export_retrieval_query_library_yaml(output_path: str, scenarios: List[ScenarioDefinition] = None) -> str:
    """Save retrieval query library to YAML."""
    lib = build_retrieval_query_library(scenarios)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(lib, f, default_flow_style=False, sort_keys=False)
    return output_path
