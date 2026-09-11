"""Tests for data ingestion and source registry validation."""
import pytest
from pathlib import Path
from src.ingestion.source_registry import SourceRegistry
from src.ingestion.load_project_data import ProjectDataLoader
from src.ingestion.load_external_data import ExternalDataLoader

def test_source_registry_loads():
    reg = SourceRegistry("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/configs/source_registry.yaml")
    sources = reg.list_approved_sources()
    assert len(sources) >= 4
    assert "PROJECT_STAGE1" in sources
    assert "PROJECT_STAGE3" in sources

def test_source_registry_rejects_unregistered():
    reg = SourceRegistry("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/configs/source_registry.yaml")
    with pytest.raises(KeyError):
        reg.get_source("UNREGISTERED_SOURCE_XYZ")

def test_ingest_project_data_creates_raw_copies():
    reg = SourceRegistry("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/configs/source_registry.yaml")
    loader = ProjectDataLoader(reg)
    res = loader.ingest_stage1_data()
    assert Path(res["dest_path"]).exists()
    assert res["row_count"] > 1000
    assert len(res["file_hash"]) == 64

def test_external_data_loading():
    reg = SourceRegistry("C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/configs/source_registry.yaml")
    ext_loader = ExternalDataLoader(reg)
    nccn = ext_loader.load_nccn_biomarkers()
    assert "biomarkers" in nccn["data"]
    fda = ext_loader.load_fda_drugs()
    assert "drugs" in fda["data"]
    ctcae = ext_loader.load_ctcae_categories()
    assert "organ_systems" in ctcae["data"]
