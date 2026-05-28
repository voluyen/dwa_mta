#! /bin/bash
set -eo pipefail   # no -u: conda activate references unbound vars internally

ENV_NAME="dwa_mta"

BASE_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Initialize conda for this shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export TOKENIZERS_PARALLELISM=false

cd "${BASE_PATH}"

echo "Re-running span_dwa_kd_opt (GPU 6) and dwa_kd_opt (GPU 4)..."
bash scripts/opt/span_dwa_kd_opt.sh &
bash scripts/opt/dwa_kd_opt.sh
