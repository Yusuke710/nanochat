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
# Pre-download Stage 1 datasets first (full network bandwidth)

echo "Downloading Stage 1 datasets..."
python -c "
from tasks.finevision import FineVision
FineVision('olmOCR-mix-0225-documents', prompt='Free OCR.', start=12000)
FineVision('olmOCR-mix-0225-documents', prompt='Free OCR.', stop=12000)
FineVision('olmOCR-mix-0225-books', prompt='Free OCR.', start=800)
FineVision('olmOCR-mix-0225-books', prompt='Free OCR.', stop=800)
print('Stage 1 datasets ready!')
"

# -----------------------------------------------------------------------------
# Pre-download Stage 2 datasets in background while Stage 1 trains
# Uses the same task classes as vis_mid_train.py to stay in sync

echo "Starting Stage 2 dataset download in background..."
python -c "
# Prefetch Stage 2 datasets (mirrors vis_mid_train.py TaskMixture)
from tasks.finevision import FineVision
from tasks.smoltalk import SmolTalk
from tasks.mmlu import MMLU
from tasks.gsm8k import GSM8K

print('Downloading Stage 2 vision datasets...')
FineVision('LLaVA_Instruct_150K', prompt='Describe this image in detail.', start=8000)
FineVision('LLaVA_Instruct_150K', prompt='Describe this image in detail.', stop=8000)

print('Downloading Stage 2 text datasets...')
SmolTalk(split='train')
SmolTalk(split='test')
MMLU(subset='auxiliary_train', split='train')
MMLU(subset='all', split='test', stop=5200)
GSM8K(subset='main', split='train')
GSM8K(subset='main', split='test', stop=420)

print('Stage 2 dataset downloads complete!')
" &
STAGE2_DOWNLOAD_PID=$!

# -----------------------------------------------------------------------------
# Number of processes/GPUs to use
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# Stage 1 and 2 training epochs (configurable)
STAGE1_EPOCHS=${STAGE1_EPOCHS:-1}
STAGE2_EPOCHS=${STAGE2_EPOCHS:-1}
SAM_GRADIENT_CHECKPOINTING=${SAM_GRADIENT_CHECKPOINTING:-False}

# -----------------------------------------------------------------------------
# Stage 1: Vision Token Training
# Train vision encoder (SAM + CLIP + projector) with LLM decoder

echo "=============================================="
echo "Stage 1: Vision Token Training (${STAGE1_EPOCHS} epoch(s))"
echo "=============================================="

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.vis_tok_train -- \
    --num_epochs=$STAGE1_EPOCHS \
    --sam_gradient_checkpointing=$SAM_GRADIENT_CHECKPOINTING \
    --run=$WANDB_RUN

# -----------------------------------------------------------------------------
# Stage 2: Vision Mid Training
# Freeze SAM, train CLIP + projector + fresh GPT on mixed vision/text data

# Wait for Stage 2 dataset downloads to complete
echo "Waiting for Stage 2 dataset downloads to complete..."
wait $STAGE2_DOWNLOAD_PID
echo "Stage 2 datasets ready!"

echo "=============================================="
echo "Stage 2: Vision Mid Training (${STAGE2_EPOCHS} epoch(s))"
echo "=============================================="

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.vis_mid_train -- \
    --resume_from_deepencoder=checkpoints/deepencoder_stage1.pt \
    --num_epochs=$STAGE2_EPOCHS \
    --run=$WANDB_RUN

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
