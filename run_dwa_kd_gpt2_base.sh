set -e

ENV_NAME="dwa_mta"

# Initialize conda for this shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "Re-running dwa_kd_gpt2_base (the one that failed)..."
bash scripts/gpt2/dwa_kd_gpt2_base.sh
