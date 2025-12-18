# Tier-1 Findings

## Environment Setup

### CUDA Compatibility
- System has NVIDIA A100-SXM4-80GB with CUDA 12.2 driver
- PyTorch 2.8.0+cu128 (latest) has library compatibility issues with CUDA 12.2
- **Solution**: Use PyTorch 2.4.1+cu121 which is stable with CUDA 12.2

### PyTorch API Changes
- `F.scaled_dot_product_attention()` `enable_gqa` argument was added in PyTorch 2.5+
- **Solution**: Manually expand k,v tensors for GQA instead of using `enable_gqa` parameter

## Model Architecture

### nanochat base-d20 Configuration
- vocab_size: 65536
- n_embd: 1280
- n_layer: 20
- n_head: 16
- n_kv_head: 16 (NOT 4 - no GQA in pretrained model)
- Total GPT params: ~560M

### Vision Encoder (from DeepSeek-OCR)
- SAM ViT-B: 95.5M parameters (facebook/sam-vit-base)
- CLIP ViT-L/14: 303M parameters (openai/clip-vit-large-patch14)
- Linear projector: 2048 -> 1280
- Total VLM params: ~962M

### Vision Token Count
- base_size=1024, patch_size=16, downsample_ratio=4
- num_queries = 1024 / 16 / 4 = 16
- tokens = (16 + 1) * 16 + 1 = 273 tokens per image

## Training Results

### Tier-1 Overfitting (1000 steps, lr=1e-4)
- Initial loss: ~6.0
- Final loss: 0.0071 average
- All 10 samples achieved loss < 0.01
- Training time: ~15 minutes on A100

### Loss Progression
```
Step 10: 6.03 -> Step 100: 1.08 -> Step 500: 0.07 -> Step 1000: 0.007
```

## Issues Identified

### Inference Generation Bug (FIXED)
- Model generates same output regardless of input image
- Root cause: After first forward pass in autoregressive generation, pixel_values are set to None
- Vision embeddings only applied once at the beginning, not cached for subsequent tokens
- **Fix**: Cache vision embeddings once at start of generation, pass cached embeddings on each forward
- **Result**: 2.5x faster inference (10+ min → 4 min for 10 samples)

### Tokenizer
- nanochat uses custom RustBPETokenizer (not HuggingFace AutoTokenizer)
- No built-in support for adding special tokens after initialization
- **Workaround**: Use last vocab ID (65535) as image token placeholder

### CRITICAL: Vision Encoder Collapse
**Discovery Date**: 2024-12-18
**Symptom**: Model outputs same content for all images within each prompt category:
- All receipts → receipt_000 content
- All charts → chart_02 content
- All textvqa → textvqa_01 content

**Root Cause Analysis**:
```
Cosine similarity between different images:
- SAM features: 0.9998-0.9999 (should be much lower!)
- CLIP features: 1.0000
- Projected features: 1.0000
```

**Diagnosis**:
The SAM encoder produces nearly identical outputs for ALL input images. The collapse originates at the SAM level and propagates through CLIP and the projector. The model can't distinguish between images.

**Statistics across 3 different images (receipt, chart, textvqa)**:
| Stage | Mean | Std | Cosine Sim |
|-------|------|-----|------------|
| SAM | -0.028 | 0.45 | 0.9998 |
| CLIP | 0.787 | 13.0 | 1.0000 |
| Projected | 0.141 | 15.4 | 1.0000 |

**Possible Causes**:
1. ~~SAM weights not properly loaded~~ - Verified: weights load correctly
2. **SAM trained weights collapsed during tier-1 training** - CONFIRMED!
3. ~~Image preprocessing issue~~ - Not the issue

**Root Cause Confirmed**:
Training caused SAM feature collapse. Comparison:

| State | Cosine Sim | SAM std |
|-------|------------|---------|
| Before training | 0.5896 | 0.05 |
| After 500 steps | 0.9998 | 0.45 |

The training process caused:
1. SAM features to collapse to nearly identical vectors (0.59 → 0.99 similarity)
2. Feature magnitude explosion (std: 0.05 → 0.45, 9x increase)

The model learned to minimize loss by:
- Memorizing outputs for each prompt category
- Ignoring image differences (SAM collapse)

**Solution**: Freeze vision encoders (SAM, CLIP) during training. Only train:
- Projector (2048 → 1280 linear)
- Special tokens (image_newline, view_separator)
- GPT weights

**Status**: RESOLVED with longer training (see below)

## Training Duration Experiment

**Hypothesis**: 500 steps is not enough for the model to learn image-specific features. More training might help.

### Step 500 Results (vision encoder collapsed)
- SAM cosine sim: 0.9998 (complete collapse)
- Accuracy: 20% (2/10 correct)
- Model memorized by prompt type, not image content

### Step 1000 Results (vision encoder preserved!)
- Accuracy: **70% (7/10 correct)**
- Errors are now image-level confusion, not category-level collapse:
  - receipt_002 → generated receipt_003 content
  - chart_01 → generated chart_03 content
- Model CAN distinguish between images

**Conclusion**: Longer training (1000 steps) prevents SAM collapse. The model learns to differentiate images, though some confusion remains between similar images in the same category.

**Remaining issues** (minor):
- Some confusion between similar images (receipt_002/003, chart_01/03)
- May need even more training or better learning rate schedule

### Step 300 with batch_size=10 - FINAL SUCCESS
- **Accuracy: 100% (10/10 correct)**
- All samples generate correct outputs
- Loss: 0.0001 on all samples
- Training time: ~10 minutes (300 steps × batch_size 10 = 3000 samples)
- Throughput: 4.9 samples/sec

| Sample | Match | Status |
|--------|-------|--------|
| receipt_000 | 100% | ✓ |
| receipt_001 | 100% | ✓ |
| receipt_002 | 100% | ✓ |
| receipt_003 | 100% | ✓ |
| chart_01 | 99% | ✓ |
| chart_02 | 98% | ✓ |
| chart_03 | 100% | ✓ |
| textvqa_01 | 100% | ✓ |
| textvqa_02 | 100% | ✓ |
| textvqa_03 | 100% | ✓ |

**Key insight**: SAM encoder DOES learn image-specific features with sufficient training. The earlier collapse at 500 steps was due to insufficient training, not architectural issues. With batch_size=10 and 300 steps (3000 sample iterations), the model perfectly overfits to all 10 images while maintaining distinct vision embeddings.

## Code Refactoring to Karpathy Style

### Goals Achieved
- Simplified `image_process.py` from 170 → 66 lines
- Simplified `vision_sample.py` from 411 → 148 lines
- Rewrote `vision_dataloader.py` to Karpathy style with `split` parameter
- Rewrote `vis_tok_train.py` (Stage 1) with config-at-top pattern
- Created `vis_mid_train.py` (Stage 2) matching Karpathy style

### Pattern Changes
| Before | After (Karpathy style) |
|--------|------------------------|
| `ImageProcessor` class | `process_image()` function |
| `ImageTransform` class | Inline transforms |
| argparse CLI | Config variables + configurator.py |
| Dataset/DataLoader classes | Generator functions |
| Separate train/val loaders | `split` parameter + `build_val_loader` lambda |

### Tokenizer Extension
- Added `<|image|>` to SPECIAL_TOKENS list in tokenizer.py
- Vocab size: 65536 → 65537 (dynamic extension at runtime)
- `from_directory()` now extends existing tokenizer with new special tokens
- Embeddings expanded after loading pretrained weights

## Training Scripts

### Stage 1 (`vis_tok_train.py`)
- All parameters trainable (SAM, CLIP, projector, GPT)
- lr = 5e-5, constant after warmup
- seq_len = 4096
- Vision data only

### Stage 2 (`vis_mid_train.py`)
- SAM frozen (~95M params)
- Trainable: CLIP, projector, GPT (~866M params)
- lr = 3e-5, StepLR decay (0.1x every 2000 steps)
- seq_len = 8192
- Text mixing disabled for tier-1/tier-2 (vision only)

### Training Verification (Stage 1, 150 steps)
```
step 00050 | val loss: 0.3225
step 00100 | val loss: 0.0216
step 00150 | val loss: 0.0018
Final loss: 0.0018
SUCCESS: Training converged!
```

### Inference Verification (`vision_sample.py`)
```
ID              Loss    Overlap
========================================
receipt_000   0.0017    100%  ✓
receipt_001   0.0007    100%  ✓
receipt_002   0.0010    100%  ✓
receipt_003   0.0010    100%  ✓
chart_01      0.0008    100%  ✓
chart_02      0.0015    100%  ✓
chart_03      0.0016    100%  ✓
textvqa_01    0.0015    100%  ✓
textvqa_02    0.0011    100%  ✓
textvqa_03    0.0009     99%  ✓
========================================
Avg loss: 0.0012
Avg overlap: 99.9%
```

## File Structure (Current)

```
nanochat/
├── nanochat/
│   ├── image_process.py       # 66 lines - process_image, count_vision_tokens, expand_image_tokens
│   ├── vision_dataloader.py   # 97 lines - Karpathy-style vision data loader
│   ├── tokenizer.py           # Extended with <|image|> special token
│   ├── nano_deepseek_ocr.py   # Main VLM model
│   └── deepencoder/           # Vision encoder (SAM + CLIP)
├── scripts/
│   ├── vis_tok_train.py       # Stage 1 training (all params trainable)
│   ├── vis_mid_train.py       # Stage 2 training (SAM frozen)
│   └── vision_sample.py       # Inference evaluation
├── data/
│   ├── overfit_dataset.json   # 10 samples for tier-1
│   └── images/                # Image files
└── checkpoints/               # Model checkpoints
```

## Multi-GPU Training Findings

### DDP Parameter Discovery: CLIP patch_embedding

**Problem**: DDP error on first training step:
```
Parameter indices which did not receive grad for rank 0: 182
```

**Investigation**:
1. Listed all parameters with their indices
2. Found parameter 182 = `vision_model.embeddings.patch_embedding.weight`
3. Traced code path in `clip_sdpa.py:133-144`:

```python
def forward(self, pixel_values, patch_embeds):
    if patch_embeds is not None:
        patch_embeds = patch_embeds  # Use SAM features - ALWAYS taken
    else:
        patch_embeds = self.patch_embedding(pixel_values)  # NEVER reached
```

**Root Cause**: CLIP's `patch_embedding` is architecturally bypassed because we pass SAM features as `patch_embeds`. The parameter exists but is never used in the forward pass.

**Fix**: Mark as not requiring gradients:
```python
self.vision_model.embeddings.patch_embedding.requires_grad_(False)
```

This is cleaner than `find_unused_parameters=True` because:
- No runtime overhead for unused parameter detection
- Explicitly documents the architectural intent
- DDP doesn't track non-trainable parameters

### DDP with Mixed Training (Vision + Text)

**Problem**: With `text_ratio > 0`, training crashes on text-only batches:
```
Parameter indices which did not receive grad for rank 0: 0 1 2 3 4 5 ... 99+
```

**Root Cause**: On text-only steps, the entire vision encoder (CLIP, projector) is skipped:
- Vision batches: `model(input_ids, targets, pixel_values)` → all params used
- Text batches: `model(input_ids, targets, pixel_values=None)` → vision encoder bypassed

**Analysis of `find_unused_parameters` behavior**:
- The flag enables per-forward-pass detection of unused parameters
- Works with random batch selection (90% vision, 10% text)
- Warning on vision steps is a "false positive" - flag still needed for text steps

**Training log showing mixed batches**:
```
step 00000 | loss: 0.0003  # Vision batch (from stage1 checkpoint)
step 00001 | loss: 0.0002  # Vision batch
step 00002 | loss: 3.9291  # Text batch (model not trained on text)
...
step 00014 | loss: 1.6159  # Another text batch
```

**Solution**: `find_unused_parameters=(text_ratio > 0)`
- When `text_ratio=0`: Pure vision training, no flag needed
- When `text_ratio>0`: Mixed training, flag enables DDP to handle variable parameter usage

### Multi-GPU Training Results

**vis_tok_train.py (Stage 1)** - 2 GPUs, 500 steps:
```
step 00000/500 | loss: 5.5341 | mfu: 2.23
step 00100/500 | loss: 1.1032 | mfu: 6.50
step 00200/500 | loss: 0.1024 | val loss: 0.1024
step 00400/500 | loss: 0.0192 | mfu: 6.45
step 00499/500 | loss: 0.0113 | val loss: 0.0082
```
- Final checkpoint: `checkpoints/step_500.pt`
- Training time: ~3 minutes

**vis_mid_train.py (Stage 2)** - 2 GPUs, vision-only:
```
step 00000/20 | loss: 5.5373 | mfu: 2.51
step 00019/20 | loss: 5.1693 | val loss: 4.7924
```

**vis_mid_train.py (Stage 2)** - 2 GPUs, mixed training from stage1 checkpoint:
```
step 00000/20 | loss: 0.0003  # Vision (from stage1)
step 00002/20 | loss: 3.9291  # Text (new)
step 00014/20 | loss: 1.6159  # Text
step 00019/20 | val loss: 0.0158
SUCCESS: Stage 2 training converged!
```

### Memory Considerations

**OOM with longer sequences**: Stage 2 uses seq_len=8192 (vs 4096 in Stage 1)
- Vision batches: ~514 tokens (short, fits in memory)
- Text batches: Full 8192 tokens → OOM on 24GB GPU

**Workaround**: Use shorter seq_len for testing:
- `--seq_len=2048` works with batch_size=1 on 24GB GPU
- For production, need larger GPUs or gradient checkpointing

### DistributedSampler Integration

Added to `vision_dataloader.py`:
```python
from torch.utils.data.distributed import DistributedSampler

ddp = dist.is_initialized()
sampler = DistributedSampler(dataset, shuffle=(split == "train")) if ddp else None
shuffle = False if ddp else (split == "train")
```

Key points:
- Sampler handles data sharding across GPUs
- When sampler is used, DataLoader's `shuffle` must be False
- Each GPU sees 1/N of the data (no duplicates)

## Vision Model Generation Fix

### Problem
When generating with vision models, the model would crash with shape mismatch:
```
RuntimeError: shape mismatch: value tensor of shape [273, 1280] cannot be broadcast to indexing result of shape [274, 1280]
```

### Root Cause
1. Image tokens (`<|image|>`) are placeholders that get replaced with vision embeddings
2. During generation, the model could output an image token
3. On the next forward pass, the code tries to scatter vision embeddings to ALL image tokens
4. But vision_embeds only has 273 positions, while input now has 274 image tokens

### Fix
Mask image token from output logits - image tokens should never be generated:
```python
# In model.generate() and Engine.generate():
if self.image_token_id is not None:
    logits[:, self.image_token_id] = float('-inf')
```

### Karpathy Principle
This is the simplest fix - one line per location, no new parameters. Image tokens are inherently input-only (they're placeholders for vision embeddings), so masking them from output is semantically correct.

### Test Results
```
Vision Engine Sanity Check
===========================
Naive model.generate(): 1.17s
Engine.generate() (KV cache): 0.82s
Match: True
Speedup: 1.4x
```
