#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing DWA-MTA dependencies ==="

pip install \
    "torch>=2.0.0" \
    "transformers>=4.40.0" \
    "deepspeed>=0.10.0" \
    "peft>=0.5.0" \
    "accelerate>=0.27.0" \
    "datasets" \
    "rouge-score" \
    "nltk" \
    "sentencepiece>=0.1.99" \
    "protobuf" \
    "tqdm" \
    "huggingface_hub" \
    "jsonlines" \
    "editdistance" \
    "spacy>=3.7.0" \
    "numba" \
    "wandb" \
    "evaluate" \
    "scikit-learn" \
    "scipy"

echo "=== Downloading NLTK data ==="
python -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
" || true

echo "=== Downloading spaCy model ==="
python -m spacy download en_core_web_sm

echo "=== Setting TOKENIZERS_PARALLELISM to avoid fork deadlock ==="
export TOKENIZERS_PARALLELISM=false

echo "=== Installation complete ==="
