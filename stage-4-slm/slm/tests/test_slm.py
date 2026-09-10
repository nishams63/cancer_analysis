"""
Unit Tests for Stage 4 SLM Module.
Tests dataset formatting, prompt construction, clinical decision engine triad extraction,
inference pipeline execution, and LoRA training orchestration.
"""

import os
from pathlib import Path
import pytest
import pandas as pd
import sys

SLM_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SLM_SRC_DIR))

from dataset import format_instruction_prompt, format_completion, ClinicalInstructionDataset
from model import ClinicalDecisionSupportEngine
from inference import ClinicalInferencePipeline
from train_lora import LoRATrainingPipeline


@pytest.fixture
def mock_dataframe():
    data = [
        {
            "patient_id": "PT-001",
            "note_id": "DOC-001",
            "clinical_note": "Stage IV Lung Cancer confirmed driver mutation in EGFR. Received Erlotinib 150 mg. No acute toxicities.",
            "instruction": "Analyze the clinical note and provide the patient's Risk, Key Finding, and Action.",
            "target_risk": "Low toxicity risk on Erlotinib 150 mg therapy.",
            "target_key_finding": "EGFR driver alteration confirmed; patient received Erlotinib at 150 mg; developed no acute adverse toxicities.",
            "target_action": "Continue standard clinical monitoring.",
            "split": "TRAIN"
        },
        {
            "patient_id": "PT-002",
            "note_id": "DOC-002",
            "clinical_note": "Colorectal cancer KRAS mutation receiving Docetaxel 100 mg. Severe acute dyspnea and grade 4 hypoxia.",
            "instruction": "Analyze the clinical note and provide the patient's Risk, Key Finding, and Action.",
            "target_risk": "Critical acute pulmonary toxicity risk on Docetaxel 100 mg.",
            "target_key_finding": "KRAS driver alteration confirmed; patient received Docetaxel at 100 mg; exhibits severe acute pulmonary toxicity.",
            "target_action": "Immediately hold Docetaxel and transfer to intensive care.",
            "split": "VALIDATION"
        }
    ]
    return pd.DataFrame(data)


def test_prompt_formatting():
    """Verify prompt formatting and instruction wrapping."""
    p = format_instruction_prompt("Test Instruction", "Note Text")
    assert "[INST]" in p
    assert "[/INST]" in p
    assert "Clinical Note:\nNote Text" in p

    c = format_completion("Risk X", "Finding Y", "Action Z")
    assert "Risk: Risk X" in c
    assert "Key Finding: Finding Y" in c
    assert "Action: Action Z" in c


def test_clinical_instruction_dataset(mock_dataframe):
    """Verify dataset items and split handling."""
    ds_train = ClinicalInstructionDataset(mock_dataframe, split="TRAIN")
    assert len(ds_train) == 1
    item = ds_train[0]
    assert "prompt_text" in item
    assert "completion_text" in item
    assert "full_text" in item
    assert item["patient_id"] == "PT-001"


def test_decision_engine_element_extraction():
    """Verify entity and toxicity extraction from clinical notes."""
    engine = ClinicalDecisionSupportEngine()
    note_crit = "Patient with KRAS mutation given Docetaxel 100 mg. Experienced acute respiratory distress and severe dyspnea."
    el = engine.extract_key_elements(note_crit)

    assert el["gene"] == "KRAS"
    assert el["drug"] == "Docetaxel"
    assert el["dosage"] == "100 mg"
    assert el["is_critical"] is True
    assert el["hazard_type"] == "PULMONARY"


def test_decision_engine_triad_generation():
    """Verify triad fields are properly structured and non-empty."""
    engine = ClinicalDecisionSupportEngine()
    note = "Breast cancer with BRCA1 driver mutation treated with Paclitaxel 80 mg/m2. No acute toxicities."
    triad = engine.generate_triad(note)

    assert "target_risk" in triad
    assert "target_key_finding" in triad
    assert "target_action" in triad
    assert "Paclitaxel" in triad["target_risk"]
    assert "Paclitaxel" in triad["target_key_finding"]
    assert len(triad["target_action"]) > 10


def test_inference_pipeline_single_and_batch():
    """Verify inference pipeline single and batch predictions."""
    pipeline = ClinicalInferencePipeline()
    note = "NSCLC EGFR mutation receiving Erlotinib 150 mg. Tolerating therapy without adverse events."
    res = pipeline.predict_note(note, note_id="N-101", patient_id="P-202")

    assert res["note_id"] == "N-101"
    assert res["patient_id"] == "P-202"
    assert "target_risk" in res
    assert "target_action" in res
    assert res["latency_ms"] >= 0.0

    batch = [
        {"clinical_note": note, "note_id": "N-1", "patient_id": "P-1"},
        {"clinical_note": note, "note_id": "N-2", "patient_id": "P-2"}
    ]
    batch_res = pipeline.predict_batch(batch)
    assert len(batch_res) == 2


def test_lora_training_cycle(tmp_path):
    """Verify LoRA training pipeline initializes, configures adapter, and produces artifacts."""
    config_content = """
model:
  name: "test-slm"
  base_model: "test/base"
  max_seq_length: 512
lora:
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules: ["q_proj", "v_proj"]
paths:
  train_data: "stage-4-slm/data-engineer/data/slm_finetune_dataset_v1.parquet"
  output_dir: "{out_dir}"
  adapter_dir: "{adapter_dir}"
""".format(out_dir=str(tmp_path / "out").replace("\\", "/"), adapter_dir=str(tmp_path / "adapter").replace("\\", "/"))

    cfg_file = tmp_path / "test_config.yaml"
    cfg_file.write_text(config_content)

    pipeline = LoRATrainingPipeline(config_path=str(cfg_file))
    res = pipeline.run_training_cycle(max_epochs=1, debug_samples=10)

    assert res["status"] == "CONVERGED"
    assert res["final_loss"] < res["initial_loss"]
    assert (tmp_path / "adapter" / "adapter_config.json").exists()
    assert (tmp_path / "out" / "training_metrics.json").exists()
