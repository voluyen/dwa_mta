#! /bin/bash
# Rerun only the OPT job on GPU 0,1.
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

# ── 2. Launch OPT job ────────────────────────────────────────────────────────
log "Launching span_dwa_kd_opt on GPU 0,1 port 7650..."
cd "${BASE_PATH}"
bash "${BASE_PATH}/scripts/opt/span_dwa_kd_opt.sh"
log "Done."
