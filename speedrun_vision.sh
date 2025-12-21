#!/bin/bash

# Vision training speedrun: Train vision encoder and evaluate on OCR benchmarks.
# This script trains a vision-language model in two stages following the DeepSeek-OCR approach.

# 1) Example launch (simplest):
# bash speedrun_vision.sh
# 2) Example launch in a screen session:
# screen -L -Logfile speedrun_vision.log -S speedrun_vision bash speedrun_vision.sh
# 3) Example launch with wandb logging:
# WANDB_RUN=vision screen -L -Logfile speedrun_vision.log -S speedrun_vision bash speedrun_vision.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Tokenizer (VLM outputs text tokens, so we need the tokenizer)
# Download pretrained tokenizer from HuggingFace instead of building from Rust

python -c "
from huggingface_hub import hf_hub_download
import shutil, os
os.makedirs('tokenizer', exist_ok=True)
path = hf_hub_download('nanochat-students/base-d20', 'tokenizer.pkl')
shutil.copy(path, 'tokenizer/tokenizer.pkl')
print('Tokenizer downloaded to tokenizer/tokenizer.pkl')
"

# Alternative: Build tokenizer from Rust source (commented out)
# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# source "$HOME/.cargo/env"
# uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# -----------------------------------------------------------------------------
# Number of processes/GPUs to use
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# Stage 1 and 2 training steps (configurable)
STAGE1_STEPS=${STAGE1_STEPS:-300}
STAGE2_STEPS=${STAGE2_STEPS:-5000}

# -----------------------------------------------------------------------------
# Stage 1: Vision Token Training
# Train vision encoder (SAM + CLIP + projector) with LLM decoder

echo "=============================================="
echo "Stage 1: Vision Token Training (${STAGE1_STEPS} steps)"
echo "=============================================="

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.vis_tok_train -- \
    --steps=$STAGE1_STEPS \
    --run=${WANDB_RUN}_stage1

# -----------------------------------------------------------------------------
# Stage 2: Vision Mid Training
# Freeze SAM, train CLIP + projector + fresh GPT on mixed vision/text data

echo "=============================================="
echo "Stage 2: Vision Mid Training (${STAGE2_STEPS} steps)"
echo "=============================================="

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.vis_mid_train -- \
    --resume_from_deepencoder=checkpoints/deepencoder_${STAGE1_STEPS}.pt \
    --steps=$STAGE2_STEPS \
    --run=${WANDB_RUN}_stage2

# -----------------------------------------------------------------------------
# Evaluation on Full OCR benchmarks

echo "=============================================="
echo "Evaluating on Fox benchmark"
echo "=============================================="
python -m scripts.vision_eval --task fox --step $STAGE2_STEPS

echo "=============================================="
echo "Evaluating on OmniDocBench benchmark"
echo "=============================================="
python -m scripts.vision_eval --task omnidocbench --step $STAGE2_STEPS

echo "=============================================="
echo "Vision training complete!"
echo "=============================================="
