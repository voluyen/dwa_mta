#! /bin/bash
# Sequential training runs — all jobs use GPU 0 one at a time.
#
# Order:
#  1. span_dwa_kd_gpt2_base
#  2. span_dwa_kd_tinyllama
#  3. span_dwa_kd_opt
#  4. span_dwa_kd_gpt2xl
#  5. dwa_kd_gpt2_medium
#  6. dwa_kd_gpt2xl
#  7. dwa_kd_opt
#  8. dwa_kd_tinyllama
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

# ── 4. Run 8 training jobs sequentially ──────────────────────────────────────
SCRIPTS=(
    "scripts/gpt2/span_dwa_kd_gpt2_base.sh"
    "scripts/tinyllama/span_dwa_kd_tinyllama.sh"
    "scripts/opt/span_dwa_kd_opt.sh"
    "scripts/gpt2xl/span_dwa_kd_gpt2xl.sh"
    "scripts/gpt2_medium/dwa_kd_gpt2_medium.sh"
    "scripts/gpt2xl/dwa_kd_gpt2xl.sh"
    "scripts/opt/dwa_kd_opt.sh"
    "scripts/tinyllama/dwa_kd_tinyllama.sh"
)

TOTAL=${#SCRIPTS[@]}
for i in "${!SCRIPTS[@]}"; do
    SCRIPT="${SCRIPTS[$i]}"
    log "[$(( i + 1 ))/${TOTAL}] Starting: ${SCRIPT}"
    bash "${BASE_PATH}/${SCRIPT}"
    log "[$(( i + 1 ))/${TOTAL}] Finished: ${SCRIPT}"
done

log "All ${TOTAL} jobs completed."
