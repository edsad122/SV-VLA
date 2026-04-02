# Speculative Verification for Vision-Language-Action Models (SV-VLA)

**Paper:** [Open-Loop Planning, Closed-Loop Verification: Speculative Verification for VLA]

---

## Overview

This repository implements **Speculative Verification for VLA (SV-VLA)**, a framework that combines efficient **open-loop long-horizon planning** with lightweight **closed-loop online verification**. 

While standard action chunking improves efficiency by predicting a sequence of future actions, it is sensitive to environmental changes and prone to error accumulation. SV-VLA addresses this by using a heavy VLA as a low-frequency **Macro-Planner** to generate an action chunk and a planning context, while a lightweight **Verifier** continuously monitors execution based on the latest observations.



### Key Features
* **Decoupled Architecture**: Plans expensively but infrequently; verifies cheaply but continuously.
* **Deviation-based Replanning**: Automatically triggers a new heavy VLA plan only when the discrepancy between the planned action and the verifier's reference action exceeds a safety threshold.
* **High Efficiency**: Achieves a **2.17x speed-up** over short-chunk baselines while maintaining high success rates.

---

## System Requirements

### Inference
* **LIBERO Sim Benchmark**: 1 GPU with ~16 GB VRAM.
* **ALOHA Robot Tasks**: 1 GPU with ~18 GB VRAM.

### Training
* **Compute**: 1-8 GPUs with 27-80 GB VRAM, depending on the desired setup.
* **Strategy**: The heavy VLA model remains frozen during verifier training, ensuring compatibility with existing pretrained models and minimizing overhead.

## Installation

See [SETUP.md](SETUP.md) for instructions on setting up the conda environment.


## Command Templates (Train & Inference)

The following templates are aligned with the current scripts:

* Training script: [vla_scripts/train_from_pruning_tfrecord.py](vla_scripts/train_from_pruning_tfrecord.py)
* Inference/Evaluation script: [experiments/robot/libero/run_libero_eval.py](experiments/robot/libero/run_libero_eval.py)

### 1) Train Temporal-Fusion Parameters

```bash
python vla_scripts/train_from_pruning_tfrecord.py \
	--pretrained-checkpoint YOUR_MODEL_PATH_OR_HF_ID \
	--tfrecord-glob "YOUR_RLDS_TFRECORD_GLOB" \
	--save-dir "YOUR_OUTPUT_DIR" \
	--epochs 5 \
	--batch-size 8 \
	--lr 3e-4 \
	--full-num-actions-chunk 64 \
	--light-num-actions-chunk 8 \
	--log-every 10
```

Example `tfrecord-glob` format:

```text
path/to/modified_libero_rlds/libero_object_no_noops/1.0.0/libero_object-train.tfrecord-*-of-00032
```

### 2) Run LIBERO Inference/Evaluation (SV-VLA)

```bash
python experiments/robot/libero/run_libero_eval.py \
	--pretrained_checkpoint YOUR_MODEL_PATH_OR_HF_ID \
	--task_suite_name libero_goal \
	--num_open_loop_steps 64 \
	--center_crop True \
	--use_temporal_fusion True \
	--light_num_actions_chunk 8 \
	--controller_deviation_threshold 0.5 \
	--resume_checkpoint YOUR_TEMPORAL_FUSION_CKPT
```

### 3) Baseline Inference (No Temporal Fusion)

```bash
python experiments/robot/libero/run_libero_eval.py \
	--pretrained_checkpoint YOUR_MODEL_PATH_OR_HF_ID \
	--task_suite_name libero_goal \
	--num_open_loop_steps 64 \
    --center_crop True \
	--use_temporal_fusion False
```

Notes:

* Replace all `YOUR_*` placeholders with your own paths/ids.
* For temporal-fusion inference, `resume_checkpoint` (or `light_resume_checkpoint`) is required by config validation.
* If needed, explicitly set `--unnorm_key` to match your dataset statistics key.


---

## Experimental Results (LIBERO Benchmark)

SV-VLA restores the robustness lost in long-horizon open-loop execution without sacrificing significant computational advantages.

| Method | Chunk Size ($K$) | Success Rate (%) | Speed-up |
| :--- | :---: | :---: | :---: |
| BASE (Short Chunk) | 8 | 96.0% | 1.00x |
| BASE (Long Chunk) | 64 | 79.5% | **3.15x** |
| Speculative Decoding | 4 | 81.7% | 1.36x |
| **SV-VLA (Ours)** | **64** | **90.9%** | **2.17x** |

*Note: Results are averaged across LIBERO-Goal, LIBERO-Object, and LIBERO-Spatial suites. SV-VLA improves the average success rate by 11.4% compared to the open-loop $K=64$ baseline.*

---

