#! /bin/bash
# Full pipeline: conda activate → install deps → download models → train
#
# GPU allocation (6x H200):
#   GPU 0     → gpt2-medium  (340M,  Qwen1.5_1.8B_SFT_Dolly,        Full FT)
#   GPU 1,2   → tinyllama    (1.1B,  Mistral7B_Dolly_SFT,            LoRA)
#   GPU 3,4   → gpt2-xl      (1.5B,  Qwen2.5-7B-Instruct-Dolly-SFT, LoRA)
#   GPU 5     → opt-2.7b     (2.7B,  Qwen2.5-7B-Instruct-Dolly-SFT, LoRA)
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

# Prevent tokenizer fork deadlock (must be set in this shell, not a subshell)
export TOKENIZERS_PARALLELISM=false

# ── 2. Install dependencies ───────────────────────────────────────────────────
log "Installing dependencies via install.sh..."
bash "${BASE_PATH}/install.sh"

# ── 3. Download models ────────────────────────────────────────────────────────
log "Downloading models via download_model.sh..."
cd "${BASE_PATH}"
bash "${BASE_PATH}/download_model.sh"

# ── 4. Launch training jobs (each script runs torchrun in background) ─────────
log "Launching all 4 training jobs..."

bash "${BASE_PATH}/scripts/gpt2_medium/dwa_kd_gpt2_medium.sh"
log "  ✓ gpt2-medium  → GPU 0,   port 6601"

bash "${BASE_PATH}/scripts/tinyllama/dwa_kd_tinyllama.sh"
log "  ✓ tinyllama    → GPU 1-2, port 6602"

bash "${BASE_PATH}/scripts/gpt2xl/dwa_kd_gpt2xl.sh"
log "  ✓ gpt2-xl      → GPU 3-4, port 6603"

bash "${BASE_PATH}/scripts/opt/dwa_kd_opt.sh"
log "  ✓ opt-2.7b     → GPU 5,   port 6604"

log "All jobs launched. Monitor logs:"
log "  tail -f ${BASE_PATH}/outputs/gpt2/gpt2-medium/dwa_kd_gpt2_medium/*/train.log"
log "  tail -f ${BASE_PATH}/outputs/tinyllama/tinyllama-1.1b-3T/dwa_kd_tinyllama/*/train.log"
log "  tail -f ${BASE_PATH}/outputs/gpt2/gpt2-xl/dwa_kd_gpt2xl/*/train.log"
log "  tail -f ${BASE_PATH}/outputs/opt/opt-2.7b/dwa_kd_opt/*/train.log"
