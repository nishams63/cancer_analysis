"""RAG Query Builder integrating scenario intents and patient facts."""
import os
import yaml
from typing import Dict, Any, List


class RAGQueryBuilder:
    def __init__(self, query_library_path: str = None):
        self.library_path = query_library_path or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "prompts", "retrieval_query_library.yaml")
        self._intents: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.library_path):
            with open(self.library_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                for item in data.get("intents", []):
                    sid = item.get("scenario_id")
                    if sid:
                        self._intents[sid] = item

    def build_query(self, scenario: Dict[str, Any], patient: Dict[str, Any]) -> Dict[str, Any]:
        sid = scenario.get("scenario_id", "")
        intent = self._intents.get(sid, {}) or scenario.get("rag_intent", {})

        primary_query = intent.get("primary_query", "")
        if not primary_query:
            # Fallback construct
            cancer = patient.get("demographics", {}).get("cancer_type", "oncology")
            muts = " ".join(patient.get("mutations", []))
            primary_query = f"{cancer} {muts} clinical practice guideline treatment resistance"

        search_terms = intent.get("search_terms", [])
        if not search_terms:
            search_terms = patient.get("mutations", []) + [patient.get("demographics", {}).get("cancer_type", "")]

        return {
            "scenario_id": sid,
            "query_string": primary_query,
            "search_terms": search_terms,
            "required_keywords": intent.get("required_keywords", []),
            "forbidden_keywords": intent.get("forbidden_keywords", []),
            "min_documents": intent.get("min_documents", 1),
            "target_domain": intent.get("target_domain", "guidelines")
        }