"""
Stage 3 Clinical Audio Ingestion & Multi-Format Normalization Layer.
Standardizes clinical audio dictations (WAV, MP3, M4A, FLAC, OGG) to 16 kHz,
16-bit mono PCM WAV for downstream speech processing.
Includes quality gating: duration validation, silence detection, clipping detection, and SNR estimation.

NOTE: STT layer remains strictly gated for clinical note generation pending
domain-specific acoustic/vocabulary adaptation.
"""

from dataclasses import dataclass
from pathlib import Path
import subprocess
import wave
from typing import Optional, Tuple
import numpy as np
import imageio_ffmpeg


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2  # 16-bit (2 bytes)

MIN_DURATION_SECONDS = 0.5
MAX_DURATION_SECONDS = 300.0  # 5 minutes
MIN_RMS_ENERGY = 10.0  # Silence threshold for 16-bit audio
MAX_CLIPPING_RATIO = 0.05  # >5% samples at peak is excessive clipping

# Operational STT Gate
STT_GATED: bool = True
STT_GATE_REASON: str = (
    "Production STT pipeline is gated pending domain-adapted acoustic/language models. "
    "Unadapted general Windows Speech API demonstrated 63.46% WER on complex oncology terminology."
)


class AudioIngestionError(Exception):
    """Base exception for audio ingestion errors."""
    pass


class UnsupportedAudioFormatError(AudioIngestionError):
    """Raised when the audio format is not supported."""
    pass


class CorruptAudioError(AudioIngestionError):
    """Raised when the audio file cannot be decoded or transcoded."""
    pass


class AudioDurationError(AudioIngestionError):
    """Raised when audio length violates operational clinical bounds."""
    pass


class SilentAudioError(AudioIngestionError):
    """Raised when the audio signal contains only silence or near-zero energy."""
    pass


@dataclass
class AudioIngestionResult:
    input_path: str
    output_path: str
    duration_seconds: float
    sample_rate: int
    channels: int
    sample_width: int
    rms_energy: float
    snr_db: float
    clipping_ratio: float
    is_valid: bool
    stt_gated: bool = STT_GATED


def get_ffmpeg_binary() -> str:
    """Retrieve bundled static FFmpeg binary path."""
    return imageio_ffmpeg.get_ffmpeg_exe()


def validate_input_path(input_path: Path) -> None:
    """Validate that the input file exists, is non-empty, and has a supported extension."""
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {input_path}")
    if input_path.stat().st_size == 0:
        raise CorruptAudioError(f"Audio file is 0 bytes (empty): {input_path}")
    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UnsupportedAudioFormatError(
            f"Unsupported format '{input_path.suffix}'. Supported: {sorted(list(SUPPORTED_EXTENSIONS))}"
        )


def transcode_to_wav(input_path: Path, output_path: Path) -> None:
    """
    Transcode input audio to 16 kHz mono 16-bit PCM WAV using FFmpeg.
    """
    ffmpeg_exe = get_ffmpeg_binary()
    cmd = [
        ffmpeg_exe,
        "-y",  # Overwrite output
        "-i", str(input_path.resolve()),
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        "-vn",  # Discard video if container has it
        str(output_path.resolve())
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        if proc.returncode != 0:
            raise CorruptAudioError(f"FFmpeg transcoding failed (code {proc.returncode}): {proc.stderr}")
    except Exception as e:
        if isinstance(e, CorruptAudioError):
            raise
        raise CorruptAudioError(f"Subprocess execution failed while decoding audio: {str(e)}") from e


def analyze_and_validate_wav(wav_path: Path) -> Tuple[float, float, float, float]:
    """
    Inspect converted WAV file and compute signal quality metrics.
    Returns: (duration_seconds, rms_energy, snr_db, clipping_ratio)
    """
    try:
        with wave.open(str(wav_path), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)
    except Exception as e:
        raise CorruptAudioError(f"Failed to read normalized WAV structure: {str(e)}") from e

    # Validate structural invariants
    if channels != TARGET_CHANNELS:
        raise CorruptAudioError(f"Expected {TARGET_CHANNELS} channel, got {channels}")
    if framerate != TARGET_SAMPLE_RATE:
        raise CorruptAudioError(f"Expected {TARGET_SAMPLE_RATE} Hz, got {framerate}")
    if sample_width != TARGET_SAMPLE_WIDTH:
        raise CorruptAudioError(f"Expected {TARGET_SAMPLE_WIDTH} bytes/sample, got {sample_width}")

    duration = n_frames / float(framerate)
    if duration < MIN_DURATION_SECONDS:
        raise AudioDurationError(
            f"Audio duration {duration:.2f}s is shorter than minimum allowable {MIN_DURATION_SECONDS}s"
        )
    if duration > MAX_DURATION_SECONDS:
        raise AudioDurationError(
            f"Audio duration {duration:.2f}s exceeds maximum allowable {MAX_DURATION_SECONDS}s"
        )

    # Convert to 16-bit signed numpy array
    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float64)
    if len(samples) == 0:
        raise CorruptAudioError("Audio contains 0 decoded samples.")

    # 1. RMS Energy
    rms = float(np.sqrt(np.mean(samples ** 2)))
    if rms < MIN_RMS_ENERGY:
        raise SilentAudioError(
            f"Audio signal is near-complete silence (RMS = {rms:.2f} < threshold {MIN_RMS_ENERGY})"
        )

    # 2. Clipping Ratio
    clipping_count = np.sum(np.abs(samples) >= 32760)
    clipping_ratio = float(clipping_count / len(samples))

    # 3. SNR Estimate (signal power vs 10th percentile energy floor in 100ms frames)
    frame_size = int(framerate * 0.1)  # 100ms
    if len(samples) >= frame_size:
        n_chunks = len(samples) // frame_size
        frame_energies = [
            np.mean(samples[i * frame_size:(i + 1) * frame_size] ** 2)
            for i in range(n_chunks)
        ]
        noise_power = np.percentile(frame_energies, 10) + 1e-6
        signal_power = np.percentile(frame_energies, 90) + 1e-6
        snr_db = float(10.0 * np.log10(signal_power / noise_power))
    else:
        snr_db = 20.0  # Default estimate for short clips

    return duration, rms, snr_db, clipping_ratio


def ingest_audio(input_file: str | Path, output_dir: Optional[str | Path] = None) -> AudioIngestionResult:
    """
    End-to-end ingestion pipeline:
    1. Validates input format and presence.
    2. Transcodes to standard 16 kHz mono WAV.
    3. Verifies audio structure and signal quality.
    4. Returns AudioIngestionResult.
    """
    input_path = Path(input_file)
    validate_input_path(input_path)

    if output_dir is None:
        out_dir = input_path.parent / "normalized"
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_wav_path = out_dir / f"{input_path.stem}_16k_mono.wav"

    # Transcode
    transcode_to_wav(input_path, output_wav_path)

    # Validate signal
    duration, rms, snr_db, clipping_ratio = analyze_and_validate_wav(output_wav_path)

    return AudioIngestionResult(
        input_path=str(input_path.resolve()),
        output_path=str(output_wav_path.resolve()),
        duration_seconds=round(duration, 3),
        sample_rate=TARGET_SAMPLE_RATE,
        channels=TARGET_CHANNELS,
        sample_width=TARGET_SAMPLE_WIDTH,
        rms_energy=round(rms, 2),
        snr_db=round(snr_db, 2),
        clipping_ratio=round(clipping_ratio, 4),
        is_valid=True,
        stt_gated=STT_GATED
    )
