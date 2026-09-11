import pytest
import tempfile
import os
import sys
from pathlib import Path

# Ensure Integration Engineer is on path
integ_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(integ_root))

from src.storage.result_store import ResultStore
from src.integration.orchestrator import MasterOrchestrator

@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

@pytest.fixture
def result_store(temp_db):
    return ResultStore(temp_db)

@pytest.fixture
def mock_scenario_data():
    return {
        "patient": {
            "scenario_id": "PROMPT-R01",
            "clinical_archetype": "Rare Acquired Bypass",
            "mutations": ["EGFR L858R", "MET Amplification"],
            "biomarkers": {"ctdna_vaf": 0.22, "tmb": 14.5}
        },
        "narrative": {
            "narrative_text": "Patient with EGFR L858R and acquired MET amplification. Toxicity grade 3 noted."
        }
    }
