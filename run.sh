#! /bin/bash
# GPU allocation
#
#  GPU 4 (~28 GB):  span_dwa_kd_gpt2_base_phrase_level  (GPT2-base Full FT, batch=32×1)
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

# ── 4. Launch training job ────────────────────────────────────────────────────
log "Launching training job..."

bash "${BASE_PATH}/scripts/ablation/span_dwa_kd_gpt2_base_phrase_level.sh" &
log "  GPU 4 | span_dwa_kd_gpt2_base_phrase_level  port 7680"

log "Job launched. Waiting for completion..."
wait
log "Done."
