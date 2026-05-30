#! /bin/bash
# 2-round parallel training across 4 GPUs (4-7).
#
# Round 1 (parallel):
#  GPU 4 — span_dwa_kd_gpt2_base
#  GPU 5 — span_dwa_kd_tinyllama
#  GPU 6 — span_dwa_kd_opt
#  GPU 7 — span_dwa_kd_gpt2xl
#
# Round 2 (parallel, starts after round 1 finishes):
#  GPU 4 — dwa_kd_gpt2_medium
#  GPU 5 — dwa_kd_gpt2xl
#  GPU 6 — dwa_kd_opt
#  GPU 7 — dwa_kd_tinyllama
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

# ── 4. Round 1: 4 jobs in parallel ───────────────────────────────────────────
log "Round 1/2 — launching 4 jobs in parallel..."

bash "${BASE_PATH}/scripts/gpt2/span_dwa_kd_gpt2_base.sh" &
log "  GPU 0 | span_dwa_kd_gpt2_base   (PID $!)"

bash "${BASE_PATH}/scripts/tinyllama/span_dwa_kd_tinyllama.sh" &
log "  GPU 1 | span_dwa_kd_tinyllama   (PID $!)"

bash "${BASE_PATH}/scripts/opt/span_dwa_kd_opt.sh" &
log "  GPU 2 | span_dwa_kd_opt         (PID $!)"

bash "${BASE_PATH}/scripts/gpt2xl/span_dwa_kd_gpt2xl.sh" &
log "  GPU 3 | span_dwa_kd_gpt2xl      (PID $!)"

log "Round 1/2 — waiting for all 4 jobs to finish..."
wait
log "Round 1/2 — done."

# ── 5. Round 2: 4 jobs in parallel ───────────────────────────────────────────
log "Round 2/2 — launching 4 jobs in parallel..."

bash "${BASE_PATH}/scripts/gpt2_medium/dwa_kd_gpt2_medium.sh" &
log "  GPU 0 | dwa_kd_gpt2_medium      (PID $!)"

bash "${BASE_PATH}/scripts/gpt2xl/dwa_kd_gpt2xl.sh" &
log "  GPU 1 | dwa_kd_gpt2xl           (PID $!)"

bash "${BASE_PATH}/scripts/opt/dwa_kd_opt.sh" &
log "  GPU 2 | dwa_kd_opt              (PID $!)"

bash "${BASE_PATH}/scripts/tinyllama/dwa_kd_tinyllama.sh" &
log "  GPU 3 | dwa_kd_tinyllama        (PID $!)"

log "Round 2/2 — waiting for all 4 jobs to finish..."
wait
log "Round 2/2 — done. All 8 jobs completed."
