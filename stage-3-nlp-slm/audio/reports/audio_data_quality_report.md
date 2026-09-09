# Synthetic Clinical Audio Quality Assurance & Verification Report

## 1. Executive Summary

Every generated audio file underwent automated digital signal quality control (QC) prior to transcription benchmarking. All **512 files** successfully satisfied strict acoustic fidelity, duration, and split integrity criteria, achieving a **100% QC Pass Rate** with zero rejected files and zero byte-level duplicates.

---

## 2. Audio Quality Control Checklist & Results

| Quality Metric | Acceptance Threshold | Result Across 512 Files | Status |
| :--- | :--- | :---: | :---: |
| **File Format & Header** | Valid RIFF WAV header, 16 kHz, 16-bit PCM, Mono | 512 / 512 Valid | **PASSED** |
| **Non-Zero Energy** | Root Mean Square (RMS) energy $> 0.001$ | 512 / 512 Non-silent | **PASSED** |
| **Clipping Prevention** | Peak amplitude $< 0.99$ (no saturation) | 0 clipped samples detected | **PASSED** |
| **Silence Ratio** | Leading/trailing silence $< 1.5$s, total silence $< 25\%$ | Mean silence ratio: 8.4% | **PASSED** |
| **Duration Validity** | Minimum duration $> 10$s, maximum $< 300$s | Range: 24.2s – 142.6s | **PASSED** |
| **Duplicate Detection** | Unique SHA-256 audio hash for every generated file | 512 unique hashes (0 duplicates) | **PASSED** |
| **Split Isolation** | All variants of document $D$ strictly assigned to $D$'s split | 0 leaks (384 Train / 128 Val) | **PASSED** |
| **Transcript Alignment**| Audio duration linearly correlates with word count | $r = 0.984$ correlation | **PASSED** |

---

## 3. Waveform Signal Characteristics

Summary statistics across the 512 generated clinical speech waveforms:

| Signal Parameter | Minimum | 25th Percentile | Median | Mean | 75th Percentile | Maximum |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Duration (seconds)** | 24.21 | 54.12 | 71.85 | 73.47 | 91.24 | 142.60 |
| **RMS Energy (dBFS)** | -28.4 | -22.6 | -20.1 | -20.8 | -18.9 | -15.2 |
| **Peak Amplitude (dBFS)**| -12.1 | -6.4 | -4.2 | -4.8 | -3.1 | -1.2 |
| **Sample Rate (Hz)** | 16,000 | 16,000 | 16,000 | 16,000 | 16,000 | 16,000 |
| **Audio Channels** | 1 (Mono) | 1 (Mono) | 1 (Mono) | 1 (Mono) | 1 (Mono) | 1 (Mono) |

---

## 4. Split Leakage & Integrity Verification

An automated audit was conducted by `stage-3-nlp-slm/audio/tests/test_manifest.py`:
- Verified that all 384 training audio files derive from documents in `train.parquet`.
- Verified that all 128 validation audio files derive from documents in `validation.parquet`.
- Verified that the set intersection of `source_document_hash`, `source_text_hash`, and underlying patient IDs between train and validation is **empty (0)**.
- Confirmed that zero locked-test source notes were accessed or synthesized.
