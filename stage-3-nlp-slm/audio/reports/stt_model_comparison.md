# Speech-to-Text (STT) Model Comparison & Benchmark

## 1. Executive Summary

This report evaluates candidate Speech-to-Text (STT) models for transcribing synthetic oncology encounter dictations. Models were evaluated against the 128 VALIDATION audio files under identical acoustic conditions.

---

## 2. STT Benchmark Performance Matrix

| STT Model Candidate | Model Family / Backend | Execution Status | Evaluated Audio Files | Word Error Rate (WER) | Character Error Rate (CER) | Average Latency (s/file) | Real-Time Factor (RTF) | Reason / Technical Blocker |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Windows SAPI** | `System.Speech.Recognition` (MS-1033) | **EVALUATED** | **128 files** | **63.46%** | **29.40%** | **10.73 s** | **0.146** | Fully offline native Windows recognizer |
| **Whisper-Tiny** | `faster-whisper (tiny.en)` | **NOT RUN** | 0 files | NA | NA | NA | NA | `faster-whisper` package download blocked |
| **Whisper-Base** | `faster-whisper (base.en)` | **NOT RUN** | 0 files | NA | NA | NA | NA | `faster-whisper` package download blocked |
| **Whisper-Small**| `faster-whisper (small.en)` | **NOT RUN** | 0 files | NA | NA | NA | NA | `faster-whisper` package download blocked |

---

## 3. Analysis of the Windows SAPI Recognizer

### Strengths
1. **Zero External Dependency Overhead**:
   Runs entirely via native Windows .NET assemblies (`System.Speech.Recognition.SpeechRecognitionEngine`) without requiring PyTorch, CUDA, or external weights.
2. **Computational Speed**:
   Demonstrates a Real-Time Factor (RTF) of **0.146** (processes ~73.5 seconds of audio in 10.73 seconds on CPU, approximately 7x faster than real-time speech).
3. **General English Dictation**:
   High accuracy on common conversational filler phrases (*"Patient presented today for evaluation..."*, *"Follow-up in three weeks"*).

### Limitations on Clinical Narrative Speech
1. **High General Word Error Rate**:
   Overall **WER of 63.46%** across the 128 validation audio recordings.
2. **Biomedical Out-of-Vocabulary (OOV) Deficit**:
   The default acoustic and language models in `MS-1033-80-DESK` were trained on consumer dictation and lack specialized oncological terminology:
   - *"cisplatin"* $\rightarrow$ transcribed as *"this flat in"*
   - *"pemetrexed"* $\rightarrow$ transcribed as *"perimeter set"*
   - *"EGFR T790M"* $\rightarrow$ transcribed as *"easy of our T seven nine zero and"*
   - *"mg/m2"* $\rightarrow$ transcribed as *"milligrams per meter squared"* or dropped.

---

## 4. Benchmark Conclusion

The native offline recognizer successfully establishes a real empirical baseline. However, its high WER (63.5%) directly impairs downstream concept extraction and triage classification, underscoring the critical need for specialized clinical acoustic/language models before audio ingestion can be considered viable for clinical oncology systems.
