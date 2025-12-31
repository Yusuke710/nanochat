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

## Quick Start: Full Training (Phase 1 + Phase 2)

Train vision encoder from scratch with a single command:

```bash
bash speedrun_vision.sh
```

This script handles everything:
1. Sets up Python environment with uv
2. Downloads tokenizer and datasets
3. **Phase 1**: Trains net_2, net_3, projector on LLaVA_Instruct_150K (frozen: SAM core, CLIP, GPT)
4. **Phase 2**: Trains projector + fresh GPT on mixed vision + text (frozen: ALL SAM, CLIP)
5. Evaluates on ChatCORE (language), Fox, and OmniDocBench (vision)

**Config options** (environment variables):
```bash
NPROC_PER_NODE=8                  # Number of GPUs (default: 8)
PHASE1_EPOCHS=1                   # Phase 1 epochs (default: 1)
PHASE2_EPOCHS=1                   # Phase 2 epochs (default: 1)
WANDB_RUN=my-run                  # WandB run name (default: dummy = no logging)
```

**Example with logging:**
```bash
WANDB_RUN=vision-run bash speedrun_vision.sh
```

---

## Quick Start: Phase 2 Only (Skip Phase 1)

Use a pretrained DeepEncoder from HuggingFace:

```bash
# Download pretrained DeepEncoder
DEEPENCODER_PATH=$(python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('Yusuke710/nano-deepencoder', 'deepencoder_stage1.pt'))
")

# Run Phase 2 (multi-GPU)
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

### Phase 1 (vis_tok_train.py) - LLaVA Stage 1 style
```bash
--steps=1000              # Number of training steps
--num_epochs=1            # Or use epochs instead of steps
--micro_batch_size=4      # Batch size per GPU (total_batch_size=256)
--lr=1e-3                 # Learning rate (LLaVA Stage 1)
--eval_every=100          # Evaluate every N steps
--save_every=100          # Save checkpoint every N steps
--run=wandb-run-name      # WandB run name (dummy = no logging)
```

### Phase 2 (vis_mid_train.py) - LLaVA Stage 2 style
```bash
--resume_from_deepencoder=...   # Path to Phase 1 checkpoint (REQUIRED)
--steps=5000                    # Number of training steps
--num_epochs=1                  # Or use epochs instead of steps
--micro_batch_size=4            # Batch size per GPU (total_batch_size=128)
--lr=2e-5                       # Learning rate (LLaVA Stage 2)
--seq_len=4096                  # Sequence length
--resume_step=1000              # Resume from Phase 2 checkpoint
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
| Phase 1, batch=4, seq=4096 | ~20GB | SAM frozen, much lower memory |
| Phase 1, batch=8, seq=4096 | ~35GB | A100 80GB OK |
| Phase 2, batch=4, seq=4096 | ~25GB | All vision frozen |
| Phase 2, batch=8, seq=4096 | ~40GB | A100 80GB OK |

**OOM fixes:**
- Reduce `--micro_batch_size`
- Reduce `--seq_len=2048`
- Use more GPUs with torchrun

---

## Upload Models to HuggingFace

```bash
# Upload DeepEncoder (Phase 1)
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('Yusuke710/nano-deepencoder', exist_ok=True, private=True)
api.upload_file('checkpoints/deepencoder_stage1.pt', 'deepencoder_stage1.pt', 'Yusuke710/nano-deepencoder')
"

# Upload full model (Phase 2)
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
