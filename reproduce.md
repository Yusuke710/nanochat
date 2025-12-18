# Reproduce Instructions

Quick setup guide for rented GPU environments.

## 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart terminal
```

## 2. Clone and Setup

```bash
git clone <your-repo-url> nanochat
cd nanochat
uv sync
```

## 3. Download Tokenizer

```bash
uv run python -c "
from huggingface_hub import hf_hub_download
import shutil, os
os.makedirs('tokenizer', exist_ok=True)
path = hf_hub_download('nanochat-students/base-d20', 'tokenizer.pkl')
shutil.copy(path, 'tokenizer/tokenizer.pkl')
print('Tokenizer downloaded to tokenizer/tokenizer.pkl')
"
```

## 4. Prepare Data

For tier-1 overfitting (10 samples):
```bash
# Data should already be in data/train.json, data/val.json, data/images/
ls data/
```

If missing, create from overfit_dataset.json:
```bash
cp data/overfit_dataset.json data/train.json
cp data/overfit_dataset.json data/val.json
```

---

## Stage 1: Vision Token Training (vis_tok_train.py)

All parameters trainable (SAM, CLIP, projector, GPT).

### Single GPU
```bash
uv run python -m scripts.vis_tok_train --steps=500 --batch_size=1
```

### Multi-GPU (recommended)
```bash
uv run torchrun --standalone --nproc_per_node=2 -m scripts.vis_tok_train \
    --steps=500 \
    --batch_size=1
```

### Config options
```bash
--steps=500          # Number of training steps
--batch_size=1       # Batch size per GPU (increase if GPU memory allows)
--lr=5e-5            # Learning rate
--eval_every=100     # Evaluate every N steps
--save_every=100     # Save checkpoint every N steps
--resume_step=100    # Resume from checkpoint (e.g., step_100.pt)
```

### Expected output
```
step 00000/500 | loss: 5.5341 | mfu: 2.23
step 00100/500 | loss: 1.1032 | val loss: 1.0xxx
step 00200/500 | loss: 0.1024 | val loss: 0.1xxx
step 00499/500 | loss: 0.01xx | val loss: 0.00xx
Final checkpoint saved to checkpoints/step_500.pt
```

---

## Stage 2: Vision Mid Training (vis_mid_train.py)

SAM frozen, trains CLIP + projector + GPT. Supports mixed vision + text training.

### Prerequisites for mixed training (text_ratio > 0)

```bash
# Install pyarrow
uv pip install pyarrow

# Download FineWeb text data (2 shards minimum)
uv run python -m nanochat.dataset -n 2 -w 2

# Copy tokenizer for text dataloader
mkdir -p ~/.cache/nanochat/tokenizer
cp tokenizer/tokenizer.pkl ~/.cache/nanochat/tokenizer/tokenizer.pkl
```

### Single GPU - Vision only
```bash
uv run python -m scripts.vis_mid_train \
    --steps=100 \
    --batch_size=1 \
    --text_ratio=0.0 \
    --resume_from_stage1=checkpoints/step_500.pt
```

### Multi-GPU - Vision only
```bash
uv run torchrun --standalone --nproc_per_node=2 -m scripts.vis_mid_train \
    --steps=100 \
    --batch_size=1 \
    --text_ratio=0.0 \
    --resume_from_stage1=checkpoints/step_500.pt
```

### Multi-GPU - Mixed training (90% vision, 10% text)
```bash
uv run torchrun --standalone --nproc_per_node=2 -m scripts.vis_mid_train \
    --steps=100 \
    --batch_size=1 \
    --text_ratio=0.1 \
    --seq_len=2048 \
    --resume_from_stage1=checkpoints/step_500.pt
```

### Config options
```bash
--steps=5000              # Number of training steps
--batch_size=4            # Batch size per GPU
--lr=3e-5                 # Learning rate (lower than stage 1)
--seq_len=8192            # Sequence length (use 2048 for 24GB GPUs)
--text_ratio=0.1          # 10% text, 90% vision (0.0 = vision only)
--resume_from_stage1=...  # Path to stage 1 checkpoint
--resume_step=1000        # Resume from stage 2 checkpoint
--lr_decay_step=2000      # Decay LR every N steps
--lr_decay_gamma=0.1      # LR decay factor
```

### Expected output (from stage1 checkpoint)
```
Loading from stage 1 checkpoint: checkpoints/step_500.pt
Freezing SAM encoder...
Parameters: 866,191,616 trainable / 962,362,880 total

step 00000/100 | loss: 0.0003 | lr: 3.00e-07  # Low loss from stage1
step 00002/100 | loss: 3.9291 | lr: 9.00e-07  # Text batch (higher loss)
...
step 00099/100 | val loss: 0.01xx
SUCCESS: Stage 2 training converged!
```

---

## Inference / Evaluation

```bash
uv run python -m scripts.vision_sample \
    --checkpoint=checkpoints/step_500.pt \
    --data_dir=data
```

### Expected output
```
ID              Loss    Overlap
========================================
receipt_000   0.0017    100%  ✓
receipt_001   0.0007    100%  ✓
...
========================================
Avg loss: 0.00xx
Avg overlap: 99.x%
```

---

## Engine Test (KV Cache)

Test that naive generation matches KV-cached Engine generation:

```bash
uv run python << 'EOF'
import time, torch
from pathlib import Path
from contextlib import nullcontext
from PIL import Image
from nanochat.common import compute_init, autodetect_device_type
from nanochat.tokenizer import RustBPETokenizer
from nanochat.gpt import GPTConfig
from nanochat.nano_deepseek_ocr import build_nano_deepseek_ocr
from nanochat.image_process import process_image, count_vision_tokens, expand_image_tokens
from nanochat.engine import Engine

ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

tokenizer = RustBPETokenizer.from_directory("tokenizer")
image_token_id = tokenizer.encode_special("<|image|>")

ckpt_path = str(max(Path("checkpoints").glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1])))
print(f"Loading {ckpt_path}")

gpt_config = GPTConfig(sequence_len=4096, vocab_size=tokenizer.get_vocab_size(),
                       n_layer=20, n_head=16, n_kv_head=16, n_embd=1280)
model = build_nano_deepseek_ocr(gpt_config=gpt_config)
model.set_image_token_id(image_token_id)
model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
model = model.eval().to(device)

pixel_values = process_image(Image.open("data/images/chart_01.png").convert("RGB"), base_size=1024)
pixel_values = pixel_values.unsqueeze(0).to(device)

n_img_tokens = count_vision_tokens(base_size=1024)
prompt_ids = [image_token_id] + tokenizer.encode("Describe:")
expanded_ids = expand_image_tokens(prompt_ids, image_token_id, n_img_tokens)

# Naive
input_ids = torch.tensor([expanded_ids], dtype=torch.long, device=device)
t0 = time.time()
with autocast_ctx:
    output = model.generate(input_ids, pixel_values=pixel_values, max_new_tokens=32, temperature=0.0)
naive_tokens = output[0, len(expanded_ids):].tolist()
naive_time = time.time() - t0

# Engine
engine = Engine(model, tokenizer)
engine_tokens = []
t0 = time.time()
with autocast_ctx:
    for tok, _ in engine.generate(expanded_ids, pixel_values=pixel_values, num_samples=1, max_tokens=32, temperature=0.0):
        engine_tokens.append(tok[0])
engine_time = time.time() - t0

print(f"Naive: {tokenizer.decode(naive_tokens)[:50]}... ({naive_time:.2f}s)")
print(f"Engine: {tokenizer.decode(engine_tokens)[:50]}... ({engine_time:.2f}s)")
print(f"Match: {naive_tokens[:len(engine_tokens)] == engine_tokens}")
print(f"Speedup: {naive_time/engine_time:.1f}x")
EOF
```

---

## Quick Test Commands

### Verify environment
```bash
uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
```

### Verify multi-GPU
```bash
uv run python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

### Verify tokenizer
```bash
uv run python -c "
from nanochat.tokenizer import RustBPETokenizer
tok = RustBPETokenizer.from_directory('tokenizer')
print(f'Vocab size: {tok.get_vocab_size()}')
print(f'Image token: {tok.encode_special(\"<|image|>\")}')
"
```

### Verify model loading
```bash
uv run python -c "
from nanochat.nano_deepseek_ocr import build_nano_deepseek_ocr
model = build_nano_deepseek_ocr()
print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')
"
```

---

## Troubleshooting

### OOM (Out of Memory)
- Reduce `--batch_size=1`
- Reduce `--seq_len=2048` (for stage 2)
- Use more GPUs with torchrun

### DDP errors
- "Parameter indices which did not receive grad": Already fixed in code
- If using `text_ratio > 0`, `find_unused_parameters=True` is auto-enabled

### Missing tokenizer
```bash
# For vision training
cp tokenizer/tokenizer.pkl tokenizer/tokenizer.pkl

# For text dataloader
mkdir -p ~/.cache/nanochat/tokenizer
cp tokenizer/tokenizer.pkl ~/.cache/nanochat/tokenizer/tokenizer.pkl
```

### Missing FineWeb data
```bash
uv run python -m nanochat.dataset -n 2 -w 2
```

---

## GPU Memory Reference

| Config | GPU Memory | Notes |
|--------|------------|-------|
| Stage 1, batch=1, seq=4096 | ~18GB | Single GPU OK |
| Stage 1, batch=2, seq=4096 | ~22GB | 24GB GPU OK |
| Stage 2, batch=1, seq=8192 | OOM | Use seq=2048 |
| Stage 2, batch=1, seq=2048 | ~20GB | 24GB GPU OK |
| Stage 2, batch=1, seq=4096 | ~22GB | 24GB GPU tight |
