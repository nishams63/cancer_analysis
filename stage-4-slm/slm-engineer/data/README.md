# Stage 5 Data Directory: Fine-Tuning Manifest & Lineage

This directory maintains immutable provenance pointers to the frozen Stage 4 fine-tuning dataset:

- **Source Path**: `stage-4-slm/data-engineer/data/slm_finetune_dataset_v1.parquet`
- **SHA-256**: `95d684c0940be3475375c69fc99a17f42b424d95ebc5a102cf608fa0889a1b2d`
- **Total Records**: 5,706
- **Total Patients**: 1,000 (0 cross-split leakage)

Per Rule 2 of the Stage 5 specification, the Stage 4 dataset is treated as strictly read-only and is not duplicated or modified. Detailed token length profiles and split metadata are recorded in `manifests/dataset_manifest.json`.
