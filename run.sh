#! /bin/bash
# GPU allocation — 4× H200 140 GB (GPUs 4-7)
#
#  GPU 4 (~45 GB):  span_dwa_kd_opt        (OPT-2.7B LoRA + Qwen2.5-7B, MTA, batch=32×1)
#
#  GPU 5 (~35 GB):  span_dwa_kd_gpt2xl                 (GPT2-XL   LoRA + Qwen2.5-7B, batch=32×1)
#                   span_dwa_kd_gpt2_base_wo_weight    (GPT2-base Full FT,            batch=32×1)
#
#  GPU 6 (~28 GB):  span_dwa_kd_gpt2_base              (GPT2-base Full FT, batch=32×1)
#                   span_dwa_kd_gpt2_base_phrase_level  (GPT2-base Full FT, batch=32×1)
#                   span_dwa_kd_gpt2_base_word_level    (GPT2-base Full FT, batch=32×1)
#
#  GPU 7 (~38 GB):  span_dwa_kd_tinyllama              (TinyLlama LoRA + Mistral-7B,  batch=32×1)
#                   span_dwa_kd_gpt2_medium             (GPT2-medium Full FT,          batch=32×1)
#
# All scripts share effective batch = 32 (batch_size × grad_acc).
#
# Usage: bash run.sh

set -eo pipefail   # no -u: conda activate references unbound vars internally

BASE_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_NAME="dwa_mta"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── 1. Conda ──────────────────────────────────────────────────────────────────
log "Activating conda env: ${ENV_NAME}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export TOKENIZERS_PARALLELISM=false

# ── 2. Install dependencies ───────────────────────────────────────────────────
log "Installing dependencies via install.sh..."
bash "${BASE_PATH}/install.sh"

# ── 3. Download models ────────────────────────────────────────────────────────
log "Downloading models via download_model.sh..."
cd "${BASE_PATH}"
bash "${BASE_PATH}/download_model.sh"

# ── 4. Launch all 8 training jobs ────────────────────────────────────────────
log "Launching 8 training jobs across 4× H200 140 GB..."

# GPU 0 — OPT-2.7B + MTA (heaviest job, runs alone)
bash "${BASE_PATH}/scripts/opt/span_dwa_kd_opt.sh" &
log "  GPU 0 | span_dwa_kd_opt        port 7650"

# GPU 1 — GPT2-XL + wo_weight ablation
bash "${BASE_PATH}/scripts/gpt2xl/span_dwa_kd_gpt2xl.sh" &
log "  GPU 1 | span_dwa_kd_gpt2xl               port 7640"

bash "${BASE_PATH}/scripts/ablation/span_dwa_kd_gpt2_base_wo_weight.sh" &
log "  GPU 1 | span_dwa_kd_gpt2_base_wo_weight  port 7660"

# GPU 2 — GPT2-base main + phrase-level + word-level ablations
bash "${BASE_PATH}/scripts/gpt2/span_dwa_kd_gpt2_base.sh" &
log "  GPU 2 | span_dwa_kd_gpt2_base              port 7610"

bash "${BASE_PATH}/scripts/ablation/span_dwa_kd_gpt2_base_phrase_level.sh" &
log "  GPU 2 | span_dwa_kd_gpt2_base_phrase_level port 7680"

bash "${BASE_PATH}/scripts/ablation/span_dwa_kd_gpt2_base_word_level.sh" &
log "  GPU 2 | span_dwa_kd_gpt2_base_word_level   port 7670"

# GPU 3 — TinyLlama + GPT2-medium
bash "${BASE_PATH}/scripts/tinyllama/span_dwa_kd_tinyllama.sh" &
log "  GPU 3 | span_dwa_kd_tinyllama  port 7630"

bash "${BASE_PATH}/scripts/gpt2_medium/span_dwa_kd_gpt2_medium.sh" &
log "  GPU 3 | span_dwa_kd_gpt2_medium port 7620"

log "All 8 jobs launched. Waiting for completion..."
wait
log "Done."
