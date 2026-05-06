# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for **DWA-MTA (Dynamic Warping Alignment - Multi-Teacher Alignment)**, a knowledge distillation framework for LLMs. The core contribution is integrating Soft-DTW (Dynamic Time Warping) alignment loss into the Dual Space KD (DSKD) framework to handle cross-tokenizer sequence alignment between student and teacher models.

## Training Commands

All training scripts live in `scripts/<model>/`. Set `BASE_PATH` to the project root before running.

**Launch training (example - GPT-2 base with DWA-KD):**
```bash
bash scripts/gpt2/dwa_kd_gpt2_base.sh
```

**Launch training (TinyLlama with DWA-KD, multi-GPU):**
```bash
bash scripts/tinyllama/dwa_kd_tinyllama.sh
```

The scripts use `torchrun` with DeepSpeed. The entry point is always:
```bash
torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/code/distillation.py ${OPTS}
```

**Run evaluation across benchmarks (dolly, self-inst, vicuna, sinst):**
```bash
bash scripts/eval/run_eval.sh <checkpoint_path> [batch_size]
```

**Single eval run:**
```bash
bash scripts/eval/eval_main.sh <device> <port> <n_gpus> <work_dir> <ckpt_path> <dataset> <batch_size> <seed>
```

Required environment variables before running:
```bash
export PYTHONPATH=${BASE_PATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUMBA_CUDA_MAX_COMPUTE_CAPABILITY=8.0   # set inside distillation.py
```

## Architecture

### Core Training Flow

`code/distillation.py` is the main training script. It:
1. Parses args from `code/arguments.py`
2. Initializes a `Distiller` (student + teacher model wrapper)
3. Builds a criterion via `criterions.build_criterion(args)`
4. Trains using DeepSpeed + distributed PyTorch

### Key Components

**`code/distiller.py` — `Distiller(nn.Module)`**
Wraps student, teacher, and optional original model. Manages:
- Loading HuggingFace models with optional LoRA (PEFT)
- Cross-vocab token/ID mappings for mismatched tokenizers
- Projector networks (MLP layers) for bridging student/teacher hidden spaces, configured via JSON (`configs/projector_config.json`)

**`code/criterions/` — Loss functions**

| Criterion name (--criterion) | Class | Description |
|---|---|---|
| `cross_entropy` | `CrossEntropyLoss` | Baseline SFT |
| `various_divergence` | `VariousDivergence` | KD with configurable divergence (forward_kl, reverse_kl, js, skewed_*) |
| `dual_space_kd` | `DualSpaceKD` | DSKD baseline |
| `dual_space_kd_with_cma` | `DualSpaceKDWithCMA` | DSKD + CMA attention alignment |
| `dwa_kd` | `DWAKD` | **Main contribution**: DSKD + CMA + Soft-DTW alignment loss |
| `universal_logit_distillation` | `UniversalLogitDistillation` | ULD baseline |
| `min_edit_dis_kld` | `MinEditDisForwardKLD` | MinEdit baseline |
| `self_correction_dskd` | `SelfCorrectionDSKD` | Self-correction variant |

**`code/criterions/soft_dtw_cuda.py` — `SoftDTW`**
CUDA-accelerated Soft-DTW implementation (wraps Numba CUDA kernels). Has a Numba `cuCtxGetDevice` patch at module load to fix broken CUDA builds. Sequence length must be ≤ 1024 for CUDA; falls back to CPU otherwise.

Key API:
- `SoftDTW(use_cuda, gamma, normalize, bandwidth)` — `forward(X, Y)` or `forward_with_cost_matrix(C, return_alignment=True)`
- `return_alignment=True` returns the row-normalized alignment matrix A = ∂sDTW/∂C

### DWA-KD Loss (`DWAKD`)

Total loss = `ce_rate * CE + kd_rate * KD + dtw_rate * DTW`

The DTW loss computes normalized Soft-DTW on cosine-distance cost matrices between student/teacher hidden states and embeddings. Banding (Sakoe-Chiba-style soft penalty) can be driven by the CMA attention alignment (`--dtw-band-source cma`) or sDTW alignment itself (`--dtw-band-source sdtw`). Key hyperparameters:
- `--dtw-rate`, `--dtw-gamma`, `--dtw-gamma-start/end/steps` (linear annealing)
- `--dtw-band-width`, `--dtw-band-penalty`, `--dtw-band-entropy-coef`
- `--dtw-hidden-layers`, `--dtw-hidden-weight`, `--dtw-embed-weight`

### Data

**`code/data_utils/distill_datasets.py`** — `DistillDataset` / `SelfCorrectionDistillDataset`
Handles parallel student/teacher tokenization. Datasets are expected in `data/dolly/` (or similar) as `.jsonl` files split into `train.jsonl`, `dev.jsonl`, `test.jsonl`.

**`code/data_utils/prompt_datasets.py`** — `PromptDataset`
Used only during evaluation (`evaluate_main.py`).

### Supported Model Families

Student: `gpt2`, `gpt2-medium`, `gpt2-xl`, `opt`, `tinyllama`, `llama`, `llama2`, `llama3`, `mistral`, `minicpm`, `gemma2`, `qwen`

Teacher: same list — cross-vocabulary distillation (e.g., GPT-2 student ← Qwen teacher) requires `--teacher-to-student-id-mapping`.

## Directory Structure

```
code/           # All Python source
  arguments.py  # All CLI args (model, data, hp, gen, peft, wandb, distiller)
  distillation.py  # Main training loop
  distiller.py  # Student/teacher model container
  evaluate.py / evaluate_main.py  # Eval utilities
  criterions/   # All loss implementations
  data_utils/   # Dataset classes
  analysis/     # Analysis scripts (LLM judge, structure dist, simulation)
scripts/
  gpt2/ gpt2xl/ gpt2_medium/ opt/ tinyllama/  # Training scripts per model
  eval/         # Eval scripts
configs/        # DeepSpeed configs and projector_config.json (not in repo — user-created)
```

## Key Hyperparameters to Know

- `--criterion`: selects the loss class (see table above)
- `--kd-objective`: divergence type for KD term (`forward_kl`, `reverse_kl`, `js_divergence`, `skewed_forward_kl`, `skewed_reverse_kl`, `adaptive_kl`)
- `--peft lora`: enables LoRA on the student; `--peft-lora-r/alpha/dropout` control rank/scale
- `--projector-config-path`: JSON defining MLP bridge networks between student and teacher hidden dimensions
- DeepSpeed config selected by `--deepspeed_config` (bf16/fp16/fp32 variants)
- W&B logging enabled with `--wandb`; `--wandb-project` / `--wandb-run-name` configure the run
