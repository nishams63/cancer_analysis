# Optional synthetic clinical audio

Real offline Windows System.Speech TTS reads approved TRAIN/VALIDATION source notes.
Run `python src/audio_generation.py --documents 64` for the pilot. Windows speech
may need normal-user execution outside a restricted process sandbox. The child
PowerShell execution policy is process-local; no system policy is changed.

Three desktop voices were verified. Other registered voices failed synthesis, so
they are not counted as speakers. Eight variants include genuine TTS rate/voice
changes and seeded signal noise, pauses and volume changes. WAV files are mono
16 kHz PCM16. File hashes and source hashes allow split and duplicate verification.

Run `python src/transcription.py --model tiny.en`, `--model base.en`, and
`--model small.en` for actual optional faster-whisper evaluation. Reference text
never enters transcription. Full generation is blocked until a measured pilot
acceptance record exists; waveform QC alone does not establish clinical quality.

NER comparisons project only unchanged transcript spans into reference coordinates.
All gold spans remain in the denominator; unmappable predicted spans count as false
positives. Clinical phrase preservation and generic WER/CER are separate measures.
Corpus negation accuracy requires independent contextual labels; lexical negation
preservation is not claimed to be clinical context accuracy.

Speech fine-tuning is not currently justified without a completed pilot and an
independent natural-speech cohort. Do not describe pretrained transcription as training.
