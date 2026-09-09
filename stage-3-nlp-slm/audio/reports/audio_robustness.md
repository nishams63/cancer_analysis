# Audio Robustness & Environmental Acoustic Analysis

## 1. Executive Summary

This report analyzes transcription robustness across the 8 controlled acoustic conditions generated in the synthetic audio pilot (16 validation files per condition, 128 validation files total).

Clinical environments present diverse acoustic challenges: clinicians may speak rapidly during emergencies, dictate with hesitant pauses, or record dictations in noisy outpatient clinics or hospital wards.

---

## 2. Robustness Matrix Across 8 Acoustic Conditions

| Acoustic Condition | Word Error Rate (WER) | Character Error Rate (CER) | Dosage Preservation | Adverse Event Preservation | Average Latency (s) | Relative Degradation vs Clean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`clean_normal` (Reference)**| **54.2%** | **24.1%** | **93.8%** | **37.5%** | **9.82 s** | **Baseline (0%)** |
| **`slow_speech`** | 58.6% | 26.4% | 90.5% | 35.0% | 12.45 s | +4.4% WER |
| **`fast_speech`** | 71.8% | 34.2% | 78.6% | 22.5% | 7.91 s | +17.6% WER |
| **`mild_noise` ($\text{SNR}\approx 25\text{ dB}$)** | 62.4% | 28.7% | 85.0% | 30.0% | 10.15 s | +8.2% WER |
| **`moderate_noise` ($\text{SNR}\approx 15\text{ dB}$)**| **79.4%** | **39.8%** | **68.2%** | **17.5%** | **10.35 s** | **+25.2% WER (Hardest)**|
| **`pause_variation`** | 56.1% | 25.3% | 91.2% | 36.0% | 11.20 s | +1.9% WER |
| **`volume_variation`** | 61.2% | 28.0% | 86.4% | 31.5% | 10.42 s | +7.0% WER |
| **`speaker_variation`** | 63.9% | 28.8% | 84.5% | 25.0% | 13.50 s | +9.7% WER |

---

## 3. Acoustic Vulnerability Hierarchy

Ranking conditions from least degraded to most degraded:

1. **Pause Variation (+1.9% WER)**: Recognizer handles inter-sentence silence effectively without dropping adjacent words.
2. **Slow Speech (+4.4% WER)**: Lengthened vowel durations slightly degrade word-boundary detection, but maintain high dosage retention (90.5%).
3. **Volume Variation (+7.0% WER)**: Dynamic range scaling causes minor clipping or low-amplitude dropouts.
4. **Mild Noise (+8.2% WER)**: Speech energy remains well above Gaussian noise floor, preserving main dictation structure.
5. **Speaker Variation (+9.7% WER)**: Pitch and formant shifts across the 3 synthetic voices slightly affect acoustic feature extraction.
6. **Fast Speech (+17.6% WER)**: Compressed phoneme durations cause elision errors, dropping adverse event recall to 22.5%.
7. **Moderate Noise (+25.2% WER — Hardest Condition)**: At 15 dB SNR, background noise masks consonant fricatives and unvoiced stops, driving WER to **79.4%** and degrading dosage preservation to **68.2%**.

---

## 4. Engineering Recommendations for Audio Ingestion

- **Front-End Denoising**: Deploy spectral subtraction or a Wiener filter prior to STT to attenuate ambient clinic noise.
- **VAD Preprocessing**: Voice Activity Detection (VAD) should strip non-speech segments to prevent spurious hallucinated tokens in noisy environments.
