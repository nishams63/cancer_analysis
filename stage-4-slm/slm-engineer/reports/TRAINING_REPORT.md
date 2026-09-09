# Stage 5 Training Report: Clinical SLM Fine-Tuning Infrastructure

**Date**: 2026-09-09  
**Role**: SLM Engineer  
**Model Architecture**: `Qwen/Qwen2.5-1.5B-Instruct`  
**Fine-Tuning Method**: Parameter-Efficient Fine-Tuning (PEFT) with LoRA ($r=16, \alpha=32$)  

---

## 1. Hardware & Execution Environment

Empirically probed and verified via `src/utils.py::detect_hardware()`:

```text
Operating System:      Windows-10-10.0.19045-SP0 (AMD64)
Python Version:        3.11.16
PyTorch Version:       2.14.0+cpu
Transformers Version:  5.17.0
PEFT Version:          0.20.0
Accelerate Version:    1.15.0
CUDA Available:        False
CUDA Device:           None (CPU Execution Only)
System RAM:            15.70 GB Total, ~7.56 GB Available
Free Disk Space:       42.20 GB
Training Status:       BLOCKED (CUDA GPU REQUIRED)
```

---

## 2. Dataset Provenance & Split Profile

- **Source Path**: `stage-4-slm/data-engineer/data/slm_finetune_dataset_v1.parquet`
- **Dataset SHA-256**: `95d684c0940be3475375c69fc99a17f42b424d95ebc5a102cf608fa0889a1b2d`
- **Total Records**: 5,706
- **Unique Patients**: 1,000

### Split Distribution & Leakage Verification

| Split Name | Record Count | Percentage | Unique Patients | Cross-Split Patient Leakage | Duplicate Notes |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | 3,996 | 70.03% | 700 | **0** | 0 |
| **VALIDATION** | 849 | 14.88% | 150 | **0** | 0 |
| **TEST** | 861 | 15.09% | 150 | **0** | 0 |
| **TOTAL** | **5,706** | **100.0%** | **1,000** | **0 (Strict Isolation)** | **0** |

---

## 3. Sequence Length & Token Distribution Analysis

Measured using official fast tokenizer `Qwen2TokenizerFast` (vocab size: 151,665) on ChatML formatted inputs:

| Metric Category | Minimum Tokens | Maximum Tokens | Mean Tokens | 95th Percentile (P95) |
| :--- | :---: | :---: | :---: | :---: |
| **Input Prompt Tokens** | 156 | 335 | 238.7 | 320.0 |
| **Assistant Target Tokens** | 51 | 126 | 79.3 | 108.0 |
| **Total Sequence Length** | 217 | 453 | 318.0 | 424.0 |

- **Truncation Rate at max_seq_length = 512**: **0.00% (0 / 3,996 records)**
- **Truncation Rate at max_seq_length = 1024**: **0.00% (0 / 3,996 records)**

---

## 4. Model Architecture & LoRA Parameter Budget

- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct`
  - Total Layers: 28 transformer decoder layers
  - Hidden Dimension: 1536
  - Feed-Forward Dimension: 8960
  - Attention Heads: 12 query heads, 2 key-value heads (Grouped Query Attention)
  - Base Parameters: **1,562,179,072**
- **LoRA Configuration**:
  - Rank ($r$): 16
  - Alpha ($\alpha$): 32 (Scaling Factor: 2.000)
  - Dropout: 0.05
  - Target Modules: All 7 linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
  - Target Matrices: $28 \times 7 = 196\text{ projection weights}$
  - Trainable Parameters: **18,464,768** (~18.46M)
  - Trainable Fraction: **1.1820%**
  - Estimated Adapter Size: **~35.2 MB** (FP16/BF16)

---

## 5. Planned Training Execution Parameters (for GPU Run)

```text
Training Epochs:              3
Per-Device Batch Size:        4
Gradient Accumulation Steps:  4
Effective Batch Size:         16
Optimizer:                    AdamW (weight_decay = 0.01)
Learning Rate:                2e-4 (0.0002)
Learning Rate Scheduler:      Cosine decay with 5% warmup
Precision:                    BF16 (or FP16 on older GPUs)
Gradient Checkpointing:       Enabled
Evaluation Cadence:           Every 50 steps on VALIDATION split
Save Cadence:                 Every 50 steps (save_total_limit = 2)
Best Checkpoint Selection:    Lowest eval_loss on VALIDATION
Estimated Steps:              (3996 * 3) / 16 ≈ 750 steps
Estimated Duration:           ~45 minutes on NVIDIA T4 (16GB VRAM)
```

---

## 6. Documented Failures & Blocked Execution

### Local CPU Training Attempt
- **Failure Mode**: `TRAINING STATUS: BLOCKED (CUDA GPU REQUIRED)`
- **Root Cause**: Attempting full backpropagation across 1.56B parameters on an AMD64 CPU without AVX-512 tensor accelerators requires ~40+ hours and risks severe memory swapping on 16GB system RAM.
- **Rule 1 Compliance Resolution**: Rather than generating artificial loss values, simulating training loops, or creating 36-byte placeholder safetensors files, the pipeline safely halted and created a production-ready Colab GPU notebook (`notebooks/train_qwen_colab.ipynb`) and shell script (`run_gpu_training.sh`) for execution on real CUDA accelerators.
