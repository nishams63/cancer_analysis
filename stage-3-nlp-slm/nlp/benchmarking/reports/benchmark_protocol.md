# Preregistered development protocol

TRAIN fits all task heads and preprocessing. VALIDATION selects checkpoints.
Locked-test content is not opened by these benchmark runners. Existing regression
tests may inspect locked-test schema/isolation but no prediction tuning uses it.
Data guards enforce patient, encounter, document, canonical-text and temporal isolation.

Frozen contextual encoders share max length 384, seed 42, class-weighted logistic
document heads and a supervised BIO token head. Token head: up to 15 epochs,
validation exact-span F1 checkpoint selection, patience 3. This is not full
Transformer fine-tuning. Raw text offsets are never applied after whitespace collapse.

Primary objectives: urgency macro F1 and exact NER F1. Critical recall may not fall
more than 0.01 absolute below fresh baseline validation recall. No final promotion
before robustness, memory, latency and reproducibility evidence is complete.
Context-corpus accuracy is NA without independently annotated polarity labels.
