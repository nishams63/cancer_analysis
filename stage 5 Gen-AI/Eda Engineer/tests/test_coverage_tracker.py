"""Tests for Scenario Coverage Matrix and Versions Document."""
import os
import json
import pytest
import pandas as pd
from src.prompts.coverage_tracker import (
    generate_coverage_matrix, export_prompt_coverage_csv, generate_prompt_versions_json
)


def test_coverage_matrix_shape():
    df = generate_coverage_matrix()
    assert len(df) >= 15
    required_cols = [
        "scenario_id", "title", "category", "blind_spots",
        "target_stages", "difficulty", "clinical_severity", "cancer_type"
    ]
    for col in required_cols:
        assert col in df.columns


def test_100_percent_blind_spot_coverage():
    df = generate_coverage_matrix()
    all_blind_spots = set()
    for bs_list in df["blind_spots"]:
        for bs in bs_list.split(";"):
            all_blind_spots.add(bs.strip())

    expected_blind_spots = {f"BS{i:02d}" for i in range(1, 16)}
    assert expected_blind_spots.issubset(all_blind_spots), (
        f"Missing blind spots: {expected_blind_spots - all_blind_spots}"
    )


def test_prompt_versions_json(tmp_path):
    out_file = str(tmp_path / "test_versions.json")
    generate_prompt_versions_json(out_file)
    assert os.path.exists(out_file)
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["catalog_version"] == "1.0.0"
    assert data["blind_spots_covered_count"] == 15
    assert len(data["scenarios"]) >= 15