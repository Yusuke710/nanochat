#!/bin/bash

# Vision training speedrun: Train vision encoder and evaluate on OCR benchmarks.
# This script trains a vision-language model in two phases following the LLaVA 1.5 approach.

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
# for vastAI
source $HOME/.local/bin/env 
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
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

# Micro batch size per GPU
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}

# Total batch size
TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-$(($NPROC_PER_NODE * $MICRO_BATCH_SIZE))}

# Phase 1 and 2 training epochs (configurable)
PHASE1_EPOCHS=${PHASE1_EPOCHS:-2}
PHASE2_EPOCHS=${PHASE2_EPOCHS:-1}

# -----------------------------------------------------------------------------
# Phase 1: Alignment Training (LLaVA-style)
# Train net_2, net_3 (compression adapter) + projector with frozen SAM/CLIP/GPT

echo "=============================================="
echo "Phase 1: Alignment Training (${PHASE1_EPOCHS} epoch(s))"
echo "  - Training: net_2, net_3, projector (~8.5M params)"
echo "  - Frozen: SAM core, CLIP, GPT"
echo "  - Data: LLaVA + olmOCR documents + books"
echo "=============================================="

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.vis_tok_train -- \
    --num_epochs=$PHASE1_EPOCHS \
    --micro_batch_size=$MICRO_BATCH_SIZE \
    --total_batch_size=$TOTAL_BATCH_SIZE \
    --run=$WANDB_RUN

# -----------------------------------------------------------------------------
# Phase 2: Fine-tuning (LLaVA-style)
# Train projector + fresh GPT with frozen vision encoders

echo "=============================================="
echo "Phase 2: Fine-tuning (${PHASE2_EPOCHS} epoch(s))"
echo "  - Training: projector, GPT (~564M params)"
echo "  - Frozen: ALL SAM (incl net_2/net_3), CLIP"
echo "  - Data: 70% vision + 30% text"
echo "=============================================="

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.vis_mid_train -- \
    --resume_from_deepencoder=checkpoints/deepencoder_stage1.pt \
    --num_epochs=$PHASE2_EPOCHS \
    --micro_batch_size=$MICRO_BATCH_SIZE \
    --total_batch_size=$TOTAL_BATCH_SIZE \
    --run=$WANDB_RUN

# -----------------------------------------------------------------------------
# Language evaluation (ensure language ability is preserved)

echo "=============================================="
echo "Evaluating language ability (ChatCORE)"
echo "=============================================="
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i vis_mid

# -----------------------------------------------------------------------------
# Evaluation on Full OCR benchmarks (auto-detects latest checkpoint)

echo "=============================================="
echo "Evaluating on Fox benchmark"
echo "=============================================="
python -m scripts.vision_eval --task=fox

echo "=============================================="
echo "Evaluating on OmniDocBench benchmark"
echo "=============================================="
python -m scripts.vision_eval --task=omnidocbench

echo "=============================================="
echo "Vision training complete!"
echo "=============================================="
