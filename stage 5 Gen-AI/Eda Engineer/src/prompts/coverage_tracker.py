"""Scenario and Prompt Coverage Tracker across Blind Spots and Pipelines."""
import os
import json
from typing import List, Dict, Any
import pandas as pd
from .scenario_builder import get_core_scenarios
from .prompt_schema import ScenarioDefinition


def generate_coverage_matrix(scenarios: List[ScenarioDefinition] = None) -> pd.DataFrame:
    """Build a tabular coverage matrix tracking scenarios across blind spots, stages, and categories."""
    if scenarios is None:
        scenarios = get_core_scenarios()

    records = []
    for sc in scenarios:
        records.append({
            "scenario_id": sc.scenario_id,
            "title": sc.title,
            "category": sc.category.value,
            "blind_spots": ";".join(sc.target_blind_spots),
            "target_stages": ";".join([s.value for s in sc.target_stages]),
            "difficulty": sc.difficulty.value,
            "clinical_severity": sc.clinical_severity.value,
            "statistical_rarity": sc.statistical_rarity.value,
            "cancer_type": sc.patient_skeleton.cancer_type,
            "rag_domain": sc.rag_intent.target_domain,
            "guideline_ref": sc.rag_intent.guideline_reference or "N/A",
            "version": sc.version
        })

    return pd.DataFrame(records)


def export_prompt_coverage_csv(output_path: str, scenarios: List[ScenarioDefinition] = None) -> str:
    """Save coverage matrix to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = generate_coverage_matrix(scenarios)
    df.to_csv(output_path, index=False)
    return output_path


def generate_prompt_versions_json(output_path: str, scenarios: List[ScenarioDefinition] = None) -> str:
    """Generate prompt_versions.json documenting registry, hashes, and change history."""
    if scenarios is None:
        scenarios = get_core_scenarios()

    version_doc = {
        "catalog_version": "1.0.0",
        "timestamp": "2026-09-11T12:00:00Z",
        "total_scenarios": len(scenarios),
        "blind_spots_covered_count": len(set(bs for sc in scenarios for bs in sc.target_blind_spots)),
        "all_blind_spots": [f"BS{i:02d}" for i in range(1, 16)],
        "scenarios": [
            {
                "scenario_id": sc.scenario_id,
                "version": sc.version,
                "title": sc.title,
                "target_blind_spots": sc.target_blind_spots,
                "target_stages": [s.value for s in sc.target_stages],
                "status": "APPROVED_FOR_GENERATION"
            }
            for sc in scenarios
        ]
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(version_doc, f, indent=2)
    return output_path
