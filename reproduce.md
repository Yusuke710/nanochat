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

Uses unified multimodal pipeline with PyTorch DataLoader for high MFU:
- `TaskMixture` with `OverfitSamples` task
- `create_multimodal_loader()` with num_workers=4, pin_memory, prefetch

### Single GPU
```bash
uv run python -m scripts.vis_tok_train --steps=300 --batch_size=2
```

### Multi-GPU (recommended)
```bash
uv run torchrun --standalone --nproc_per_node=2 -m scripts.vis_tok_train \
    --steps=300 \
    --batch_size=2
```

### Config options
```bash
--steps=300          # Number of training steps
--batch_size=2       # Batch size per GPU
--lr=5e-5            # Learning rate
--eval_every=100     # Evaluate every N steps
--save_every=100     # Save checkpoint every N steps (saves DeepEncoder checkpoint)
--resume_step=100    # Resume from checkpoint (e.g., step_100.pt)
```

### Expected output
```
Train samples: 10, Val samples: 10
Batch shapes: inputs=torch.Size([2, 646]), targets=torch.Size([2, 646]), pixel_values=torch.Size([2, 3, 1024, 1024])
step 00000/300 | loss: 6.4972 | mfu: 11.70
step 00100/300 | Validation loss: 0.0169 | min: 0.0169
step 00200/300 | Validation loss: 0.0072 | min: 0.0072
step 00299/300 | Validation loss: 0.0082 | min: 0.0072
Saved DeepEncoder checkpoint to checkpoints/deepencoder_300.pt
```

Note: Stage 1 saves a DeepEncoder-only checkpoint (SAM + CLIP + projector + special tokens) for Stage 2.

---

## Stage 2: Vision Mid Training (vis_mid_train.py)

SAM frozen, trains CLIP + projector + fresh GPT (per DeepSeek-OCR paper).

Uses unified multimodal pipeline with PyTorch DataLoader:
- `TaskMixture` with `OverfitSamples` (and optionally other tasks)
- `create_multimodal_loader()` with num_workers=4, pin_memory, prefetch
- Loads DeepEncoder from Stage 1 + fresh nanochat GPT from HuggingFace

**Note:** Mixed vision+text batches not yet supported (padding issue). Currently all samples in a batch must have images OR all must be text-only.

### Single GPU
```bash
uv run python -m scripts.vis_mid_train \
    --steps=100 \
    --batch_size=2 \
    --resume_from_deepencoder=checkpoints/deepencoder_300.pt
```

### Multi-GPU (recommended)
```bash
uv run torchrun --standalone --nproc_per_node=2 -m scripts.vis_mid_train \
    --steps=100 \
    --batch_size=2 \
    --resume_from_deepencoder=checkpoints/deepencoder_300.pt
```

### Config options
```bash
--steps=5000                      # Number of training steps
--batch_size=4                    # Batch size per GPU
--lr=3e-5                         # Learning rate (lower than stage 1)
--seq_len=8192                    # Sequence length (use 2048 for 24GB GPUs)
--resume_from_deepencoder=...     # Path to DeepEncoder checkpoint (REQUIRED)
--resume_step=1000                # Resume from stage 2 checkpoint
--lr_decay_step=2000              # Decay LR every N steps
--lr_decay_gamma=0.1              # LR decay factor
```

### Configuring Task Mixture

Edit the TaskMixture section at the top of vis_mid_train.py:
```python
train_ds = TaskMixture([
    OverfitSamples(data_dir=data_dir, split="train"),  # vision task
    # SmolTalk(split="train", stop=10),  # TODO: fix mixed batch handling
])
```

### Expected output
```
Loading DeepEncoder: checkpoints/deepencoder_300.pt
Loading fresh nanochat GPT from HuggingFace...
Freezing SAM encoder...
Parameters: 866,191,616 trainable / 962,362,880 total

step 00000/100 | loss: 6.9126 | lr: 3.00e-07
step 00050/100 | loss: 1.2345 | lr: 1.50e-05
Step 00100 | Validation loss: 0.9700 | min: 0.9700
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

---

## FineVision Training with WandB (A100 80GB)

Training commands for FineVision dataset (chartqa) with WandB logging on A100 80GB.

### Setup Environment Variables

```bash
# Export .env variables for WandB and HuggingFace
set -a && source .env && set +a
```

### Stage 1: Vision Token Training (1000 steps)

```bash
source .venv/bin/activate && set -a && source .env && set +a && \
python -m scripts.vis_tok_train \
    --run=stage1-1000steps \
    --steps=1000 \
    --batch_size=10 \
    --eval_every=50 \
    --save_every=100
```

**Config:**
- Dataset: FineVision/chartqa (18K train, 100 val)
- batch_size=10, seq_len=4096
- lr=5e-5 with cosine annealing
- All components trained: SAM, CLIP, projector, GPT

**Expected results:**
- Min val loss: ~1.45 at step 850
- Time: ~13 minutes
- Output: `checkpoints/deepencoder_1000.pt`

### Stage 2: Vision Mid Training (1500 steps)

```bash
source .venv/bin/activate && set -a && source .env && set +a && \
python -m scripts.vis_mid_train \
    --run=stage2-1500steps \
    --steps=1500 \
    --batch_size=2 \
    --eval_every=100 \
    --resume_from_deepencoder=checkpoints/deepencoder_1000.pt
```

**Config:**
- Dataset: FineVision/chartqa + SmolTalk (mixed vision + text)
- batch_size=2 (batch_size=4 causes OOM with seq_len=8192)
- seq_len=8192
- lr=3e-5 with StepLR decay
- SAM frozen, trains CLIP + projector + fresh GPT

**Expected results:**
- Min val loss: ~2.16 at step 1400
- Time: ~5 minutes
- Output: `checkpoints/step_1500.pt`

### WandB Logging

Both scripts log to project `nano-deepseek-ocr`:
- Metrics: train/loss, train/lr, train/dt, train/tok_per_sec, train/mfu, train/total_flops, val/loss
- Set `--run=dummy` to disable WandB logging

### Memory Notes (A100 80GB)

| Stage | batch_size | seq_len | Status |
|-------|------------|---------|--------|
| 1 | 10 | 4096 | OK |
| 2 | 4 | 8192 | OOM |
| 2 | 2 | 8192 | OK |

### Upload Models to HuggingFace

Upload trained checkpoints to HuggingFace (private repos). Running again will update/overwrite.

**nano-deepencoder** (Stage 1 encoder-only checkpoint):
```bash
source .venv/bin/activate && set -a && source .env && set +a && python -c "
from huggingface_hub import HfApi
api = HfApi()
repo_id = 'Yusuke710/nano-deepencoder'
api.create_repo(repo_id, repo_type='model', exist_ok=True, private=True)
api.upload_file(
    path_or_fileobj='checkpoints/deepencoder_1000.pt',
    path_in_repo='deepencoder.pt',
    repo_id=repo_id,
    repo_type='model',
)
print(f'Uploaded to https://huggingface.co/{repo_id}')
"
```

**nano-deepseek-ocr** (Stage 2 full model):
```bash
source .venv/bin/activate && set -a && source .env && set +a && python -c "
from huggingface_hub import HfApi
api = HfApi()
repo_id = 'Yusuke710/nano-deepseek-ocr'
api.create_repo(repo_id, repo_type='model', exist_ok=True, private=True)
api.upload_file(
    path_or_fileobj='checkpoints/step_1500.pt',
    path_in_repo='model.pt',
    repo_id=repo_id,
    repo_type='model',
)
print(f'Uploaded to https://huggingface.co/{repo_id}')
"
```
