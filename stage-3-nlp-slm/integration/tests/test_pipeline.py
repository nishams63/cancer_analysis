"""
Unit and integration tests for the unified Stage 3 Multimodal Pipeline.
"""

import pytest
from pathlib import Path
import sys

INTEGRATION_SRC = Path(__file__).resolve().parents[1]
if str(INTEGRATION_SRC) not in sys.path:
    sys.path.insert(0, str(INTEGRATION_SRC))

from pipeline import Stage3Pipeline


def test_pipeline_text_analysis():
    pipeline = Stage3Pipeline()
    sample_text = (
        "Patient with metastatic non-small cell lung cancer harboring EGFR T790M mutation. "
        "Initiated on osimertinib 80mg daily. Currently tolerating without acute nausea or dyspnea."
    )
    result = pipeline.analyze_text(sample_text)
    
    assert result["modality"] == "TEXT"
    assert result["research_only"] is True
    assert "prediction" in result
    pred = result["prediction"]
    assert "triage_urgency" in pred
    assert "toxicity_hazard" in pred
    assert "clinical_entities" in pred
    assert pred["triage_urgency"]["predicted_class"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]


def test_pipeline_text_rejects_empty():
    pipeline = Stage3Pipeline()
    with pytest.raises(ValueError, match="Nonempty text required"):
        pipeline.analyze_text("")
    with pytest.raises(ValueError, match="Nonempty text required"):
        pipeline.analyze_text("   \n\t  ")


def test_pipeline_audio_without_stt_raises():
    pipeline = Stage3Pipeline(stt=None)
    with pytest.raises(RuntimeError, match="Optional STT backend is not configured"):
        pipeline.analyze_audio(Path("nonexistent.wav"))


def test_pipeline_audio_with_mock_stt():
    class MockSTT:
        def transcribe(self, path):
            return "Patient has EGFR mutation and received osimertinib 80mg without dyspnea."

    pipeline = Stage3Pipeline(stt=MockSTT())
    
    # Locate a real generated validation WAV file from the pilot
    audio_dir = Path(__file__).resolve().parents[2] / "audio" / "data" / "synthetic" / "validation"
    sample_wavs = list(audio_dir.glob("*.wav"))
    if not sample_wavs:
        pytest.skip("No synthetic validation audio files found to test audio pipeline.")
    
    sample_wav = sample_wavs[0]
    result = pipeline.analyze_audio(sample_wav)
    
    assert result["modality"] == "AUDIO"
    assert result["research_only"] is True
    assert "transcript" in result
    assert "normalized_transcript" in result
    assert "prediction" in result
    assert "audio_quality" in result
    assert result["audio_quality"]["qc_pass"] is True
    assert "latency_seconds" in result
    assert result["latency_seconds"]["total"] > 0
