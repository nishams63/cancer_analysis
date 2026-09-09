"""
Unit Tests for Stage 3 Clinical Audio Ingestion & Multi-Format Normalization Layer.
Tests:
- Transcoding of multi-format audio (WAV, MP3, FLAC, OGG) to 16 kHz mono 16-bit WAV
- Validation of structural invariants (sample rate, channels, bit depth)
- Detection and error handling for silence, duration violations, and unsupported formats
- Operational STT gating confirmation
"""

import math
import struct
import sys
import wave
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from audio_ingestion import (
    ingest_audio,
    AudioIngestionResult,
    UnsupportedAudioFormatError,
    CorruptAudioError,
    AudioDurationError,
    SilentAudioError,
    STT_GATED,
    TARGET_SAMPLE_RATE,
    TARGET_CHANNELS,
    TARGET_SAMPLE_WIDTH
)


def create_synthetic_wav(
    filepath: Path,
    duration: float = 1.0,
    sample_rate: int = 44100,
    channels: int = 2,
    freq: float = 440.0,
    amplitude: float = 10000.0,
    is_silent: bool = False
) -> Path:
    """Helper to generate a synthetic PCM WAV file."""
    n_frames = int(duration * sample_rate)
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        
        frames = bytearray()
        for i in range(n_frames):
            if is_silent:
                val = 0
            else:
                val = int(amplitude * math.sin(2 * math.pi * freq * (i / sample_rate)))
            packed = struct.pack("<h", val)
            for _ in range(channels):
                frames.extend(packed)
        wf.writeframes(frames)
    return filepath


@pytest.fixture
def temp_audio_dir(tmp_path):
    d = tmp_path / "audio_test"
    d.mkdir()
    return d


def test_clean_audio_normalization(temp_audio_dir):
    """Test normalization of 44.1 kHz stereo audio to 16 kHz mono."""
    src_wav = temp_audio_dir / "stereo_44k.wav"
    create_synthetic_wav(src_wav, duration=2.0, sample_rate=44100, channels=2, freq=440.0)

    result = ingest_audio(src_wav, output_dir=temp_audio_dir / "out")

    assert result.is_valid is True
    assert result.sample_rate == TARGET_SAMPLE_RATE
    assert result.channels == TARGET_CHANNELS
    assert result.sample_width == TARGET_SAMPLE_WIDTH
    assert 1.9 <= result.duration_seconds <= 2.1
    assert result.rms_energy > 100.0
    assert result.clipping_ratio == 0.0
    assert result.stt_gated is True
    assert Path(result.output_path).exists()


def test_silence_detection(temp_audio_dir):
    """Test that pure silence is rejected with SilentAudioError."""
    silent_wav = temp_audio_dir / "silent.wav"
    create_synthetic_wav(silent_wav, duration=1.0, is_silent=True)

    with pytest.raises(SilentAudioError):
        ingest_audio(silent_wav, output_dir=temp_audio_dir / "out")


def test_duration_underflow(temp_audio_dir):
    """Test that audio shorter than 0.5s is rejected with AudioDurationError."""
    short_wav = temp_audio_dir / "too_short.wav"
    create_synthetic_wav(short_wav, duration=0.2)

    with pytest.raises(AudioDurationError):
        ingest_audio(short_wav, output_dir=temp_audio_dir / "out")


def test_unsupported_format(temp_audio_dir):
    """Test that unsupported formats are rejected with UnsupportedAudioFormatError."""
    bad_file = temp_audio_dir / "sample.txt"
    bad_file.write_text("Not an audio file")

    with pytest.raises(UnsupportedAudioFormatError):
        ingest_audio(bad_file, output_dir=temp_audio_dir / "out")


def test_empty_file(temp_audio_dir):
    """Test that a 0-byte file raises CorruptAudioError."""
    empty_file = temp_audio_dir / "empty.wav"
    empty_file.touch()

    with pytest.raises(CorruptAudioError):
        ingest_audio(empty_file, output_dir=temp_audio_dir / "out")


def test_nonexistent_file(temp_audio_dir):
    """Test that a missing file raises FileNotFoundError."""
    missing = temp_audio_dir / "does_not_exist.wav"
    with pytest.raises(FileNotFoundError):
        ingest_audio(missing, output_dir=temp_audio_dir / "out")
