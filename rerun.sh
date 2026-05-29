#! /bin/bash
# Rerun only the phrase-level ablation job on GPU 4.
# Usage: bash rerun.sh

set -eo pipefail

BASE_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_NAME="dwa_mta"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── 1. Conda ──────────────────────────────────────────────────────────────────
log "Activating conda env: ${ENV_NAME}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export TOKENIZERS_PARALLELISM=false

# ── 2. Launch job ─────────────────────────────────────────────────────────────
log "Launching span_dwa_kd_gpt2_base_phrase_level on GPU 4 port 7680..."
cd "${BASE_PATH}"
bash "${BASE_PATH}/scripts/ablation/span_dwa_kd_gpt2_base_phrase_level.sh"
log "Done."
