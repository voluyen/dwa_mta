set -e
echo "Training DWA-KD for GPT2-Base..."
bash scripts/gpt2/span_dwa_kd_gpt2_base.sh

echo "Training DWA-KD for GPT2-Medium..."
bash scripts/gpt2_medium/span_dwa_kd_gpt2_medium.sh

echo "Training DWA-KD for GPT2-XL..."
bash scripts/gpt2xl/span_dwa_kd_gpt2xl.sh

echo "Training DWA-KD for TinyLlama..."
bash scripts/tinyllama/span_dwa_kd_tinyllama.sh

echo "Training DWA-KD for OPT..."
bash scripts/opt/span_dwa_kd_opt.sh
