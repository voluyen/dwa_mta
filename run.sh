set -e

echo "Training DWA-KD with word-level spans..."
bash scripts/ablation/span_dwa_kd_gpt2_base_word_level.sh

echo "Training DWA-KD with phrase-level spans..."
bash scripts/ablation/span_dwa_kd_gpt2_base_phrase_level.sh

echo "Training DWA-KD with out weight..."
bash scripts/ablation/span_dwa_kd_gpt2_base_wo_weight.sh
