# Reproduce Instructions

Quick setup guide for rented GPU environments.

## 1. Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Clone and install
git clone <your-repo-url> nanochat
cd nanochat
uv sync --extra gpu
```

## 2. Environment Variables

```bash
export HF_TOKEN=hf_xxx          # Required: HuggingFace token
export WANDB_API_KEY=xxx        # Optional: WandB logging
```

---

## Quick Start: Full Training (Stage 1 + Stage 2)

Train vision encoder from scratch with a single command:

```bash
bash speedrun_vision.sh
```

This script handles everything:
1. Sets up Python environment with uv
2. Downloads tokenizer and datasets
3. **Stage 1**: Trains SAM + CLIP + projector + GPT on OCR data
4. **Stage 2**: Freezes SAM, loads fresh GPT, trains on mixed vision + text
5. Evaluates on ChatCORE (language), Fox, and OmniDocBench (vision)

**Config options** (environment variables):
```bash
NPROC_PER_NODE=8                  # Number of GPUs (default: 8)
STAGE1_EPOCHS=1                   # Stage 1 epochs (default: 1)
STAGE2_EPOCHS=1                   # Stage 2 epochs (default: 1)
SAM_GRADIENT_CHECKPOINTING=True   # Enable for lower memory (default: False)
WANDB_RUN=my-run                  # WandB run name (default: dummy = no logging)
```

**Example with logging:**
```bash
WANDB_RUN=vision-run bash speedrun_vision.sh
```

---

## Quick Start: Stage 2 Only (Skip Stage 1)

Use a pretrained DeepEncoder from HuggingFace:

```bash
# Download pretrained DeepEncoder
DEEPENCODER_PATH=$(python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('Yusuke710/nano-deepencoder', 'deepencoder_stage1.pt'))
")

# Run Stage 2 (multi-GPU)
torchrun --standalone --nproc_per_node=8 -m scripts.vis_mid_train -- \
    --resume_from_deepencoder=$DEEPENCODER_PATH \
    --num_epochs=1

# Or single GPU
python -m scripts.vis_mid_train \
    --resume_from_deepencoder=$DEEPENCODER_PATH \
    --num_epochs=1
```

---

## Training Config Options

### Stage 1 (vis_tok_train.py)
```bash
--steps=1000              # Number of training steps
--num_epochs=1            # Or use epochs instead of steps
--batch_size=4            # Batch size per GPU
--lr=5e-5                 # Learning rate
--eval_every=100          # Evaluate every N steps
--save_every=100          # Save checkpoint every N steps
--run=wandb-run-name      # WandB run name (dummy = no logging)
```

### Stage 2 (vis_mid_train.py)
```bash
--resume_from_deepencoder=...   # Path to DeepEncoder checkpoint (REQUIRED)
--steps=5000                    # Number of training steps
--num_epochs=1                  # Or use epochs instead of steps
--micro_batch_size=4            # Batch size per GPU
--lr=3e-5                       # Learning rate (lower than stage 1)
--seq_len=4096                  # Sequence length
--resume_step=1000              # Resume from stage 2 checkpoint
--run=wandb-run-name            # WandB run name (dummy = no logging)
```

---

## Evaluation

```bash
# Vision evaluation (Fox, OmniDocBench)
python -m scripts.vision_eval --task=fox
python -m scripts.vision_eval --task=omnidocbench

# Language evaluation (ChatCORE)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i vis_mid
```

---

## Quick Test Commands

```bash
# Verify environment
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Verify tokenizer
python -c "
from nanochat.tokenizer import RustBPETokenizer
tok = RustBPETokenizer.from_directory('tokenizer')
print(f'Vocab size: {tok.get_vocab_size()}, Image token: {tok.encode_special(\"<|image|>\")}')
"

# Verify model
python -c "
from nanochat.nano_deepseek_ocr import build_nano_deepseek_ocr
model = build_nano_deepseek_ocr()
print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')
"
```

---

## GPU Memory Reference

| Config | GPU Memory | Notes |
|--------|------------|-------|
| Stage 1, batch=4, seq=4096 | ~40GB | A100 80GB OK |
| Stage 1, batch=2, seq=4096 | ~22GB | 24GB GPU OK |
| Stage 2, batch=4, seq=4096 | ~45GB | A100 80GB OK |
| Stage 2, batch=2, seq=4096 | ~25GB | 24GB GPU tight |

**OOM fixes:**
- Reduce `--batch_size` or `--micro_batch_size`
- Reduce `--seq_len=2048`
- Enable `SAM_GRADIENT_CHECKPOINTING=True`
- Use more GPUs with torchrun

---

## Upload Models to HuggingFace

```bash
# Upload DeepEncoder (Stage 1)
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('Yusuke710/nano-deepencoder', exist_ok=True, private=True)
api.upload_file('checkpoints/deepencoder_stage1.pt', 'deepencoder_stage1.pt', 'Yusuke710/nano-deepencoder')
"

# Upload full model (Stage 2)
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('Yusuke710/nano-deepseek-ocr', exist_ok=True, private=True)
api.upload_file('checkpoints/model_final.pt', 'model.pt', 'Yusuke710/nano-deepseek-ocr')
"
```

---

## Troubleshooting

### Missing tokenizer
```bash
python -c "
from huggingface_hub import hf_hub_download
import shutil, os
os.makedirs('tokenizer', exist_ok=True)
shutil.copy(hf_hub_download('nanochat-students/base-d20', 'tokenizer.pkl'), 'tokenizer/tokenizer.pkl')
"
```

### PyTorch < 2.5 compatibility
Replace `enable_gqa` in `nanochat/gpt.py`:
```python
# Instead of: y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
if self.n_head != self.n_kv_head:
    n_rep = self.n_head // self.n_kv_head
    k = k.repeat_interleave(n_rep, dim=1)
    v = v.repeat_interleave(n_rep, dim=1)
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```
