set -e

ENV_NAME="dwa_mta"
PYTHON_VERSION="3.10"

# Initialize conda for this shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create conda environment if it doesn't exist
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
else
    echo "Conda environment '${ENV_NAME}' already exists. Skipping creation."
fi

conda activate "${ENV_NAME}"

# Install dependencies
echo "Installing dependencies via install.sh..."
bash install.sh

echo "Downloading models via download_model.sh..."
bash download_model.sh

echo "Training all models in parallel..."
# echo "  GPU 0: span_dwa_kd_gpt2_base | word_level | phrase_level"
echo "  GPU 1: span_dwa_kd_gpt2_medium | dwa_kd_gpt2_base | span_dwa_kd_tinyllama"
# echo "  GPU 2: span_dwa_kd_gpt2xl | span_dwa_kd_opt"
# bash scripts/gpt2/span_dwa_kd_gpt2_base.sh &
# bash scripts/ablation/span_dwa_kd_gpt2_base_word_level.sh &
# bash scripts/ablation/span_dwa_kd_gpt2_base_phrase_level.sh &
# bash scripts/gpt2_medium/span_dwa_kd_gpt2_medium.sh &
# bash scripts/gpt2/dwa_kd_gpt2_base.sh &
# bash scripts/tinyllama/span_dwa_kd_tinyllama.sh
# bash scripts/gpt2xl/span_dwa_kd_gpt2xl.sh &
bash scripts/opt/span_dwa_kd_opt.sh
# wait
