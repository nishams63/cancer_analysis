# Stage 2 integrity audit — 2026-09-07

Status: implementation and evaluation audit, not a completed experiment report.

## Observed repository state

The working tree already contains modified baseline dataset loading and untracked upgrade modules. Preserve these contributions. Protected source/checkpoint SHA-256 values are recorded in `baseline_integrity_before.json`. No existing checkpoint is accepted solely because its filename exists.

## Findings

* Real BiLSTM optimization and independent fusion training seeds already exist and are reusable. The stated earlier arithmetic-offset implementation is not present in this checkout; do not claim to have removed code that was not observed.
* Pathology constructors silently replace failed pretrained downloads with random weights and freeze those weights. Checkpoint `pretrained=True` records the request, not evidence of loaded weights. The existing upgraded ResNet-18 checkpoint lacks dataset/seed/epoch/pretrained-source fingerprints; retain it as unverified and never use it for final comparisons.
* Undefined ROC/PR AUC defaults to 0.5, including multiclass PR-AUC which is never computed. Return undefined values explicitly and calculate multiclass PR-AUC.
* All 1,000 V2 patient bags contain benign, malignant and inflammation tiles. The any-malignant patient label is therefore constant. A successful patient-level classifier under this task provides no evidence of learning. A separate reproducible bag-composition challenge is required; future outcomes must not define pathology inputs.
* 59/1,000 last-historical rows have a missing forecast target that the loader converts to zero. Last input dates range from day 75 to 90. Targets need explicit source-day/horizon provenance and missing labels must not be fabricated.
* Inputs are filtered to <=90 days and train-only normalization exists. Manifest agreement and future-feature invariance still require stronger tests. The continuous-time embedding currently receives standardized time despite interpreting its values as days. Missing-visit perturbation does not recompute intervals or velocity.
* Attention-MIL defaults to ResNet-18 while the final report calls it ResNet-50. ResNet-50 MIL initialization also uses a one-image training-mode BatchNorm probe. Gated pooling itself is reusable.
* Ablation A relabels multiclass pathology F1 as progression F1. E/F merely copy temporal metrics despite claiming multimodal systems. H copies G before uncertainty/calibration/OOD evaluation. These are invalid comparisons even without arithmetic offsets.
* Fixed fusion stores test metrics in its validation entry. Its head calls are not under no-grad before conversion to NumPy. The report forces the ResNet-50/MIL/Transformer architecture while only choosing among learned fusion heads.
* Existing feature caches are keyed only by split and may cross-contaminate architecture/configuration experiments. New caches need source, data and weight fingerprints.
* MC dropout leaks training flags into later evaluations; binary tensor outputs use batchwise softmax. Random risk heads in learned fusion receive no loss yet are exposed as risk scores.
* OOD evaluation shifts latent embeddings directly; it is an artificial latent stress test, not evidence of detecting real OOD patient inputs. Robustness lacks explicit +/-25% changes, short histories, proper insufficient-data abstention and complete regression degradation results.
* Permutation importance clips negative changes, concealing improved performance under permutation. Frozen-backbone Grad-CAM requires input gradients. Attribution does not establish biological causation.
* Only two of five requested reports are generated. Failure analysis and API upgrade integration are absent. `/predict/batch` is not present in the current API.
* Baseline BiLSTM runs a padded bidirectional sequence without packing; backward states can depend on padding. Preserve the original source and document this limitation when benchmarking the exact baseline.

## Runtime

Python 3.13.15 is installed at the direct Python313 executable; the `py -3.13` launcher fails to discover it. PyTorch 2.14.0+cpu and torchvision 0.29.0+cpu are available. No CUDA device is available to this PyTorch build. Final CPU experiments may use deterministic cached frozen pretrained features with actual independently optimized heads. Such experiments must be described as frozen-feature transfer learning, not end-to-end fine-tuning.

## Required interpretation

Synthetic patient histology is not linked to future biomarker trajectories in the existing generator. A revised bag task can test aggregation capability but cannot manufacture evidence that pathology predicts progression. No expected architecture is guaranteed to win. No final benchmark values have been established by this audit.
