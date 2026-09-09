# Stage 4 — EDA Engineer: Instruction-Tuning Dataset Analysis

## Role Overview
The **EDA Engineer** in Stage 4 is responsible for exploratory data analysis of the SLM instruction-tuning dataset (`slm_finetune_dataset_v1.parquet`).

## Objectives & Deliverables
1. **Instruction & Note Length Profiling**:
   - Character and token length distributions across `input`, `instruction`, and targets (`target_risk`, `target_key_finding`, `target_action`).
   - Context window truncation risk analysis for target SLM architectures (e.g. 512, 1024, 2048 tokens).
2. **Entity & Vocabulary Coverage**:
   - Frequency distribution of antineoplastic drugs, genomic driver variants, and toxicities across data splits.
   - Lexical diversity and n-gram overlap between instruction inputs and target outputs.
3. **Partition Uniformity Audit**:
   - Verifying class balance, urgency, and hazard representations across Train (70%), Validation (15%), and Test (15%) partitions.
4. **Visualizations & Reports**:
   - Publication-quality figures documenting distributions, token counts, and entity density heatmaps.
