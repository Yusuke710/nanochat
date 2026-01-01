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

## DeepEncoder Checkpoint Testing

### Overview
Tested the DeepEncoder-only checkpoint feature (commit a6e5053) which saves only vision encoder weights for Stage 2 training with a fresh GPT.

### Test Sequence

**Stage 1**: Train and save DeepEncoder checkpoint
```bash
python -m scripts.vis_tok_train --steps=300 --batch_size=2
```
- Creates `checkpoints/deepencoder_300.pt` (1.6GB) - vision encoder only
- Creates `checkpoints/step_300.pt` (3.8GB) - full model

**DeepEncoder checkpoint contents verified**:
- 477 keys total
- Key prefixes: `sam_model`, `vision_model`, `projector`, `image_newline`, `view_separator`
- **No `gpt.*` keys** ✓

**Stage 2**: Load DeepEncoder + fresh GPT
```bash
python -m scripts.vis_mid_train --resume_from_deepencoder=checkpoints/deepencoder_300.pt --steps=300 --text_ratio=0.0 --batch_size=2
```
- Loads DeepEncoder from checkpoint ✓
- Downloads fresh nanochat GPT from HuggingFace ✓
- Loss decreased: 5.12 → 0.15 (val loss: 0.30) ✓

**Inference results after Stage 2 (300 steps)**:
| Sample | Loss | Overlap |
|--------|------|---------|
| receipt_000 | 0.1355 | 17% |
| receipt_001 | 0.1238 | 40% |
| receipt_002 | 0.1363 | 9% |
| receipt_003 | 0.3185 | 44% |
| chart_01 | 0.0201 | 97% ✓ |
| chart_02 | 0.0131 | 98% ✓ |
| chart_03 | 0.1844 | 26% |
| textvqa_01 | 0.0072 | 100% ✓ |
| textvqa_02 | 0.0240 | 96% ✓ |
| textvqa_03 | 0.0173 | 68% ✓ |
| **Average** | **0.098** | **59.6%** |

### Stage 1 vs Stage 2 MFU Difference

**Observation**: Stage 2 runs ~4x faster than Stage 1 despite using longer sequence length.

| Metric | Stage 1 | Stage 2 |
|--------|---------|---------|
| MFU | ~11-12% | ~40-45% |
| seq_len | 4096 | 8192 |
| batch_size | 2 | 2 |
| Trainable params | 962M (all) | 866M (90%) |

**Why the 4x speedup when only 10% params are frozen?**

Model parameter breakdown:
| Component | Params | % of Total |
|-----------|--------|------------|
| SAM encoder | 96M | 9.9% |
| CLIP vision | 303M | 31.5% |
| Projector | 2.6M | 0.3% |
| GPT | 561M | 58.3% |

The MFU difference is disproportionate because:

1. **Vision backward pass is expensive**: SAM processes 1024×1024 images with spatial attention. Backward pass through vision transformers is much more compute-intensive per parameter than text transformers due to large spatial dimensions.

2. **Activation memory**: SAM stores huge intermediate activations (high-res feature maps). Without gradients, these don't need to be retained for backprop.

3. **Different compute patterns**: Spatial attention scales O((H×W)²) which is worse than causal text attention for backward pass.

**Conclusion**: While SAM is only 10% of parameters, freezing it eliminates a disproportionate amount of compute cost, resulting in ~4x speedup.

## Two-Stage Training Results (2024-12-20)

### Configuration
- **Stage 1**: 500 steps, batch_size=4, lr=5e-5
- **Stage 2**: 1000 steps, batch_size=4, lr=3e-5
- **GPU**: Single GPU training

### Stage 1 Results (vis_tok_train.py)
- All parameters trainable (SAM, CLIP, projector, GPT)
- Initial loss: ~6.9
- Final val loss: 0.0009
- Saved: `deepencoder_500.pt` for Stage 2

### Stage 2 Results (vis_mid_train.py)
- SAM frozen, trains CLIP + projector + fresh GPT
- Loaded DeepEncoder from Stage 1 + fresh nanochat GPT from HuggingFace
- Initial loss: ~5.5
- Final val loss: ~0.02

### Inference Results (vision_sample.py with step_1000.pt)

```
ID                  Loss  Overlap
========================================
receipt_000       0.0030    100%  ✓
receipt_001       0.0145    100%  ✓
receipt_002       0.0072    100%  ✓
receipt_003       0.0060    100%  ✓
chart_01          0.0003    100%  ✓
chart_02          0.0004    100%  ✓
chart_03          0.0004    100%  ✓
textvqa_01        0.0004    100%  ✓
textvqa_02        0.0004    100%  ✓
textvqa_03        0.0004    100%  ✓
========================================
Avg loss: 0.0033
Avg overlap: 100.0%
```

**Key Achievement**: All 10 samples achieved **100% word overlap** with very low loss (0.0033 average). The two-stage training pipeline successfully overfits the model to the training samples while maintaining proper vision encoding.

## Vision Eval Metrics Verification (2024-12-22)

### Fox Benchmark Metric

**Issue Found**: Original implementation used character-level bag-of-characters precision.

**Correct Definition** (per [Fox GitHub eval_tools/eval_ocr_test.py](https://github.com/ucaslcl/Fox)):
- Uses **word-level** tokenization: `gt.split()` / `pred.split()`
- Then **set-based** precision: `len(set(gt_words) & set(pred_words)) / len(set(pred_words))`
- Paper quote: "due to the lengthy text of the document, we use word-level segmentation to calculate each indicator"

**Fix Applied**: Updated `vision_eval.py` precision/recall/f1 to use word-level set-based metrics.

### OmniDocBench Metric

**Paper Definition** ([OmniDocBench CVPR 2025](https://arxiv.org/html/2412.07626v1)):
- NED = `edit_distance(pred, gt) / max(len(pred), len(gt))` (character-level)
- Paper uses "Adjacency Search Match" for paragraph-level matching

**DeepSeek-OCR Usage** ([arxiv:2510.18234](https://arxiv.org/html/2510.18234v1)):
- Uses simple character-level edit distance
- Does NOT use paragraph matching
- Quote: "All metrics in the table are edit distances, where smaller values indicate better performance"

**Decision**: Keep current character-level NED implementation to match DeepSeek-OCR methodology.

### Summary Table

| Benchmark | Paper Metric | DeepSeek-OCR Usage | Our Implementation |
|-----------|-------------|-------------------|-------------------|
| Fox | Word-level precision | Precision (unclear level) | Word-level precision ✓ |
| OmniDocBench | NED + paragraph matching | NED only | NED only ✓ |

## Speedrun Testing (2024-12-22)

### Environment
- GPU: NVIDIA GeForce RTX 4090 (48GB VRAM)
- Storage: 300GB available

### Errors Encountered and Resolutions

1. **GPU Count Mismatch**: Script defaulted to `NPROC_PER_NODE=8` but only 1 GPU available
   - **Fix**: Set `NPROC_PER_NODE=1`

2. **WandB Authentication**: Script appended `_stage1` to `WANDB_RUN=dummy`, breaking the `run == "dummy"` check
   - **Fix**: Source `.env` file for proper WandB API key

3. **OOM (Out of Memory)**: Default `device_batch_size=10` too high for 48GB GPU
   - **Fix**: Reduce to `device_batch_size=4`

4. **Disk Space Exhaustion**: HuggingFace cache grew to 281GB, hitting 300GB limit
   - **Fix**: Clear processed datasets cache (`~/.cache/huggingface/datasets/`)

5. **CRITICAL BUG - KVCache Argument Order** (engine.py:356):
   ```python
   # WRONG: kv = KVCache(B, max_len + max_tokens, m.n_kv_head, ...)
   # CORRECT: kv = KVCache(B, m.n_kv_head, max_len + max_tokens, ...)
   ```
   - Swapped `seq_len` and `num_heads` arguments caused 225GB allocation request
   - **Fix**: Corrected argument order

6. **Checkpoint Naming**: Script expects `model_XXXXXX.pt` and `meta_XXXXXX.json` with 6-digit step number
   - **Fix**: Create proper meta.json file with model_config

### Data Download Metrics

| Dataset | Files | Time | Samples |
|---------|-------|------|---------|
| olmOCR-mix-0225-documents | 196 | ~10:28 | 228,858 |
| olmOCR-mix-0225-books | N/A | ~6s | 15,194 |
| LLaVA_Instruct_150K | 154 | ~7:17 | 157,710 (partial) |

**Total HuggingFace Cache Size**: 178GB (after cleanup)

### Vision Eval Timing

| Task | Samples | Time | Throughput |
|------|---------|------|------------|
| OmniDocBench (16 samples) | 16 | 3.0s | 5.3 samples/s |
| OmniDocBench (full) | 981 | 491.1s (8.2min) | 2.0 samples/s |
| Fox (full) | 112 | 6.6s | 17.0 samples/s |

### Benchmark Results (Pretrained Model from HuggingFace)

| Benchmark | Score | Samples | Notes |
|-----------|-------|---------|-------|
| OmniDocBench | 0.0046 | 981 | Very low - likely model undertrained or format mismatch |
| Fox | 0.0089 | 112 | Very low - 1/112 samples matched |

**DeepSeek-OCR Paper Results (for comparison)**:
| Benchmark | DeepSeek-OCR Score |
|-----------|-------------------|
| OmniDocBench | 0.148 (NED, lower is better) |
| Fox | 0.617 (Precision) |

**Note**: The pretrained model shows very low scores, suggesting either:
1. Model not fully trained
2. Output format mismatch with evaluation expectations
3. Different evaluation methodology

## OmniDocBench Eval Debugging (2024-12-23)

### Problem
Both step_4500 (Stage 2) and step_7226 (Stage 1) models produce nearly identical outputs for all images:
```
"The following is a list of the most common causes of death in the United States:
- Heart disease
- Stroke
- Cancer..."
```

### Debug Investigation

**Checked Pipeline Stages**:

| Stage | Different between images? | Notes |
|-------|---------------------------|-------|
| SAM features | Yes (slightly) | mean diff: ~0.0002, max diff: varies |
| CLIP features | Yes (slightly) | first 5 values differ |
| Projected features | Yes | max diff: ~42.9 |
| Merged embeddings | Yes | max diff: ~42.9 |
| First token logits | Yes | max diff: 0.266 |

**Key Finding**: The pipeline is working correctly - vision features ARE different between images. However:
- Top-1 token is always "The" with nearly identical probability (~9.89) for different images
- The small logit differences (~0.27 max) are not enough to change generation path

### Root Cause
**Training Issue**: The model has learned a very strong prior towards generating generic text ("The following is a list...") that dominates over the vision feature differences. The vision features exist but are too weak relative to the language model's text prior.

**Possible Causes**:
1. Vision encoder features too homogeneous (low variance between images)
2. Projector not amplifying image-specific features
3. GPT not learning to attend to vision tokens
4. Training data format/distribution mismatch

### Evidence
- Text-only prompts work correctly ("What is capital of France?" → "Paris")
- Model is not broken, just ignores vision features for generation
- Vision embeddings ARE being properly scattered into text embeddings

### Next Steps for Training
1. Increase vision encoder training (more steps, higher LR)
2. Add contrastive loss between image features
3. Freeze GPT and train only vision components initially
4. Check training data quality and prompt-response alignment

### SAM Collapse Location Identified

Tested with synthetic images (random noise, solid white, solid black) to find where embeddings become identical:

| Stage | Cosine Similarity | Status |
|-------|------------------|--------|
| **PIX (input)** | -0.007 to -1.0 | ✓ Very different |
| **SAM** | **0.9997 - 0.9999** | ❌ **COLLAPSE HERE** |
| CLIP | 0.9989 - 0.9999 | Already collapsed |
| COMBINED | 0.9989 - 0.9999 | Inherited |
| PROJECTED | 0.9996 - 0.9999 | Inherited |
| VISION_EMBEDS | 0.9996 - 0.9999 | Inherited |

**Conclusion**: The SAM encoder produces nearly identical embeddings (cosine sim > 0.999) for completely different images. The collapse originates at SAM and propagates through the entire vision pipeline.

### Default Output Source

The model outputs "The following is a list of the most common causes of death in the United States..." for ANY image input (random noise, white, black all produce identical output).

This phrase likely originates from:
1. **SmolTalk training data** - Found health-related content about "causes of death", "heart disease" in samples
2. **olmOCR PDF documents** that contain medical/health content

The model learned to ignore image content entirely and output a memorized response pattern when it sees vision tokens + "Free OCR." prompt.

### Why SAM Collapsed: Comparison with DeepSeek-OCR

**DeepSeek-OCR Paper Training** (from [arxiv:2510.18234](https://arxiv.org/html/2510.18234v1)):

| Aspect | DeepSeek-OCR | Our Training |
|--------|--------------|--------------|
| **DeepEncoder data** | 30M+ images | ~380K images |
| **Batch size** | 1280 | 128 |
| **Epochs** | 2 | ~1-2 |
| **SAM in Stage 2** | FROZEN | FROZEN |
| **Total sample iterations** | ~60M+ | ~900K |
| **Training steps** | ~47,000 | ~7,000 |

**Key insight**: DeepSeek-OCR trained SAM on **30 million images for 2 epochs with batch size 1280** before freezing it:
- ~47,000 steps of DeepEncoder training
- 60M+ sample iterations total
- Massive data diversity (30M unique images)

**Our training**:
- ~7,000 steps with batch size 128
- ~900K sample iterations
- Only 380K unique images

**Why collapse happens with limited data**:
1. **100x less data diversity** (380K vs 30M images)
2. **~70x fewer sample iterations** (900K vs 60M)
3. **Smaller batch size** affects feature learning stability

With limited data, SAM learns to output a "mean embedding" that minimizes average loss across all images rather than learning distinctive per-image features. DeepSeek's massive data diversity (30M images) forces SAM to learn generalizable visual features that distinguish different documents.

**Solutions**:
1. **Freeze SAM from the start** - use pretrained SAM weights without training
2. **Get more training data** - scale closer to DeepSeek's 30M images
3. **Train much longer** - many more epochs on current data to increase sample iterations
4. **Use pretrained vision encoder** - leverage models already trained on large-scale data

## Checkpoint Evaluation (2026-01-01)

### Checkpoints Found

| Checkpoint | Size | Stage | Step |
|-----------|------|-------|------|
| `deepencoder_stage1.pt` | 1.53 GB | Phase 1 | N/A |
| `model_000584.pt` | 3.67 GB | vis_tok_train | 584 |

### Model Structure Verification

**Stage 1 (deepencoder_stage1.pt)** - Vision Encoder Only:
- Contains: SAM model, CLIP vision model, projector
- Keys: sam_model.*, vision_model.*, projector.*
- No GPT weights (as expected for DeepEncoder-only checkpoint)

**Stage 2 (model_000584.pt)** - Full VLM:
- Contains: SAM model, CLIP vision model, projector, GPT
- Keys: sam_model.*, vision_model.*, projector.*, gpt.*
- Model config from meta_000584.json:
  - sequence_len: 4096
  - vocab_size: 65537
  - n_layer: 20
  - n_head: 16
  - n_embd: 1280

### Training Metadata

From WandB run summary:
- Total training FLOPs: 315.4 trillion
- Final train loss: 2.49
- Final val loss: 2.44
- Training step: 583
- Tokens/sec: 21,970

### Benchmark Results (step 584 checkpoint)

| Benchmark | Score | Samples | Time | Throughput |
|-----------|-------|---------|------|------------|
| **Fox** | 0.3949 | 112 | 98.1s | 1.1 samples/s |
| **OmniDocBench** | 0.0769 | 290 | 420s | 0.7 samples/s |

### Model Output Analysis

**Critical Issue**: Model generates repetitive, irrelevant outputs:
- Typical output: "The image features a large clock on the left side, with a clock on the right side..."
- Same pattern regardless of document content
- Model is NOT performing OCR - outputs generic image descriptions

**Sample Predictions**:
| Sample | Ground Truth (truncated) | Prediction |
|--------|-------------------------|------------|
| Fox #0 | "297. It was felt by most speakers..." | "The image features a large clock on the sidewalk..." |
| Fox #5 | "Cortical Specification and Neuronal Migration..." | "The image features a large clock on the left side..." |
| OmniDocBench #0 | "Copyright... McGraw-Hill Companies..." | "The image shows a man standing in front of a large white sign..." |

### Root Cause Analysis

The model exhibits classic symptoms of **vision encoder collapse** combined with **memorized generic outputs**:

1. **Vision-Language Misalignment**: The vision encoder features are not being properly translated to text generation. The projector may not be effectively bridging the vision-language gap.

2. **Generic Output Memorization**: The model has learned to output stock phrases like "The image features a large clock" instead of reading actual document content.

3. **Training Data/Stage Mismatch**: The meta file indicates stage "vis_tok_train" (Stage 1), but the checkpoint contains GPT weights. This suggests possible confusion in the training pipeline.

4. **Insufficient Training**: At only 584 steps with the current training setup, the model may not have converged on the OCR task.

### Comparison with Expected Results

| Metric | Current | DeepSeek-OCR Paper |
|--------|---------|-------------------|
| Fox (Precision) | 0.39 | ~0.62 |
| OmniDocBench (NED) | 0.08 | ~0.15 (lower is better) |

**Note**: Current OmniDocBench score of 0.08 would actually be excellent if the model were working correctly (NED is edit distance, lower is better), but the qualitative outputs confirm the model is not functioning as an OCR system.

### Recommendations

1. **Verify training pipeline**: Ensure Stage 1 and Stage 2 are properly separated
2. **Check vision encoder training**: SAM collapse may be occurring as documented earlier
3. **Increase training duration**: 584 steps may be insufficient
4. **Use pretrained DeepEncoder**: Start from a properly trained vision encoder

## Stage 2 Testing and Collapse Analysis (2026-01-01)

### Stage 2 (vis_mid_train.py) Status

**Works correctly** with reduced batch size. Default batch_size=32 causes OOM.

**Maximum batch sizes for Stage 2 (H100 93GB VRAM)**:
| Sequence Length | Max Batch Size | Memory Usage |
|-----------------|----------------|--------------|
| 1500 | 16 | 62.9 GB |
| 4096 (default) | 8 | 84.1 GB |

**Why Stage 2 needs more memory than Stage 1**:
- Stage 1: ~8.5M trainable params (net_2, net_3, projector) - GPT frozen
- Stage 2: ~564M trainable params (projector + GPT) - 66x more gradients to store

### Net_2/Net_3 Collapse Analysis

Compared feature similarity **before and after Stage 1 training**:

| Image Pair | PRETRAINED | TRAINED | Change |
|------------|------------|---------|--------|
| white vs black | -0.71 | 0.99 | +1.70 |
| white vs noise | 0.04 | 0.82 | +0.78 |
| noise vs red | 0.19 | 0.98 | +0.79 |
| black vs red | 0.55 | 0.81 | +0.25 |
| **AVERAGE** | **0.02** | **0.90** | **+0.88** |

**CONCLUSION**: Stage 1 training causes feature collapse in net_2/net_3 layers:
- Pretrained SAM produces highly distinct features (avg similarity ~0.02)
- After training, features become nearly identical (avg similarity ~0.90)
- The training optimizes for minimizing loss by collapsing to a "mean embedding"

### Feature Similarity by Layer (After Training)

| Layer | Avg Similarity | Status |
|-------|---------------|--------|
| after_neck (256ch) | 0.85 | Partial collapse |
| after_net2 (512ch) | 0.76 | Best separation |
| after_net3 (1024ch) | 0.87 | High collapse |
| vision_embeds (final) | 0.92 | Near collapse |

### Root Cause

The net_2/net_3 conv layers are trained with limited data diversity, causing them to learn features that minimize average loss across all images rather than preserving image-specific information.

### Recommendations

1. **Freeze net_2/net_3 during Stage 1** - Use pretrained weights
2. **Add contrastive loss** - Encourage distinct features between different images
3. **More data diversity** - Current 380K images insufficient vs DeepSeek-OCR's 30M
