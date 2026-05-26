set -e

ENV_NAME="dwa_mta"

# Initialize conda for this shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# These scripts use criterion=dwa_kd with GPT2/OPT student models.
# They all fail at compute_dual_space_kd_loss_with_cma() because
# GPT2 (.transformer.wte) and OPT (.model.decoder.embed_tokens) are
# not handled by the old hasattr chain.
#
# TinyLlama (span_dwa_kd_tinyllama) is NOT here — it uses
# .model.embed_tokens which the old code handled fine.
#
# Run after syncing the get_input_embeddings() fix in dwa_kd.py.

echo "=== Rerunning scripts affected by embed_tokens bug ==="

# [CONFIRMED CRASHED] — was already dead when fix was applied
echo "[1/6] dwa_kd_gpt2_base (GPU 1)..."
bash scripts/gpt2/dwa_kd_gpt2_base.sh &

# [GPT2 student — same bug, will crash on first training step if not fixed]
echo "[2/6] span_dwa_kd_gpt2_base (GPU 0)..."
bash scripts/gpt2/span_dwa_kd_gpt2_base.sh &

echo "[3/6] span_dwa_kd_gpt2_base_word_level (GPU 0)..."
bash scripts/ablation/span_dwa_kd_gpt2_base_word_level.sh &

echo "[4/6] span_dwa_kd_gpt2_base_phrase_level (GPU 0)..."
bash scripts/ablation/span_dwa_kd_gpt2_base_phrase_level.sh &

echo "[5/6] span_dwa_kd_gpt2_medium (GPU 1)..."
bash scripts/gpt2_medium/span_dwa_kd_gpt2_medium.sh &

# [GPT2-XL student — same bug]
echo "[6/6] span_dwa_kd_gpt2xl (GPU 2)..."
bash scripts/gpt2xl/span_dwa_kd_gpt2xl.sh &

# NOTE: span_dwa_kd_opt (OPT) — OPT also lacks .model.embed_tokens directly,
# but it's in run.sh too. Add here if it crashes:
# bash scripts/opt/span_dwa_kd_opt.sh &

wait
echo "=== All done ==="
