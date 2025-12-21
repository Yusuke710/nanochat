# nano-deepseek-ocr Plan

Add vision capability to nanochat using DeepSeek-OCR's vision encoder.

**Status: Tier-1 COMPLETE ✅** (2024-12-20)

## Architecture

```
Image (H×W×3)
    │
    ▼
┌─────────────────────────────────────────┐
│ SAM-ViT-B (~96M params)                 │
│ - 768 embed, 12 layers, 12 heads        │
│ - Output: (H/16, W/16, 768)             │
│ - Conv compressor: 4× reduction         │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ CLIP-L/14 (~303M params)                │
│ - Takes compressed SAM features         │
│ - 24 layers, 1024 hidden dim            │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Linear Projector (~2.6M)                │
│ - CLIP features (2048) → n_embd (1280)  │
└─────────────────────────────────────────┘
    │
    ▼
Vision Embeddings → Replace <image> positions → nanochat GPT (~561M)
```

**Total: ~962M params**

## Core Design (Karpathy Style)

1. **One tokenizer for all modalities** - `render_conversation()` handles `<image>`, `<audio>`, `<video>` via `MEDIA_PLACEHOLDERS` dict
2. **One loader for all stages** - `multimodal_data_generator()` works for vision-only and mixed training
3. **Logic in forward()** - vision embedding merge is explicit
4. **Minimal abstractions** - clear data flow, no hidden magic

## File Structure

```
nanochat/
├── nano_deepseek_ocr.py      # Main VLM model
├── deepencoder/              # Vision encoder
│   ├── sam_vary_sdpa.py
│   ├── clip_sdpa.py
│   └── load_pretrained.py
├── image_process.py          # Image → tensor
├── multimodal_dataloader.py  # Unified loader for all modalities
├── tokenizer.py              # MEDIA_PLACEHOLDERS + render_conversation()
├── gpt.py                    # nanochat GPT
├── scripts/
│   ├── vis_tok_train.py      # Stage 1 training
│   ├── vis_mid_train.py      # Stage 2 training
│   └── vision_sample.py      # Test inference
└── ...
```

## Training Stages

| Stage | Script | Trainable | Frozen | Data | Seq Len |
|-------|--------|-----------|--------|------|---------|
| 1 | `vis_tok_train.py` | SAM, CLIP, Projector, GPT (~962M) | Nothing | Vision data | 4096 |
| 2 | `vis_mid_train.py` | CLIP, Projector, GPT (~866M) | SAM (~96M) | Vision + Text | 8192 |

**Two-Stage Workflow:**
1. Stage 1 trains all params, saves `deepencoder_{steps}.pt` (vision encoder only)
2. Stage 2 loads DeepEncoder + fresh nanochat GPT from HuggingFace

**Optimizer (per DeepSeek-OCR paper):**
- Stage 1: AdamW lr=5e-5, cosine annealing
- Stage 2: AdamW lr=3e-5, StepLR decay (0.1x every 2000 steps)

## Multi-Resolution Modes

| Mode | base_size | Tokens |
|------|-----------|--------|
| Tiny | 512 | 73 |
| Small | 640 | 111 |
| **Base** | **1024** | **273** |
| Large | 1280 | 421 |
| Gundam | 1024+crops | 273+ |

Token formula: `(num_queries + 1) × num_queries + 1` where `num_queries = resolution / patch_size / downsample_ratio`

**Current:** Base mode only. Gundam mode deferred to post-Tier-2.

**Future Gundam Mode:** When Base mode works for Tier-2, add dynamic resolution with crop-based multi-scale processing. See DeepSeek-OCR paper Section 3.2.2.

## Evaluation

### Quick Testing

```bash
# Stage 1 training (all params trainable)
python -m scripts.vis_tok_train --steps=500 --batch_size=4

# Stage 2 training (SAM frozen, fresh GPT)
python -m scripts.vis_mid_train --resume_from_deepencoder=checkpoints/deepencoder_500.pt --steps=1000

# Inference evaluation
python -m scripts.vision_sample --resume_step=1000
```

### Success Criteria

| Tier | Criteria | Status |
|------|----------|--------|
| 1 | Overfit to near-zero loss on 10 images. 100% accuracy on `vision_sample.py`. | ✅ COMPLETE |
| 2 | Mixed vision+text training on scaled data (details below) | Pending |
| 3 | Competitive scores on Fox/OmniDocBench | Pending |

### Tier 1 Results (COMPLETE)

**Two-Stage Training Results:**
- Stage 1: 500 steps, batch_size=4, lr=5e-5 → val loss: 0.0009
- Stage 2: 1000 steps, batch_size=4, lr=3e-5 → val loss: 0.02

**Inference Results:**
```
ID              Loss    Overlap
========================================
receipt_000   0.0030    100%  ✓
receipt_001   0.0145    100%  ✓
receipt_002   0.0072    100%  ✓
receipt_003   0.0060    100%  ✓
chart_01      0.0003    100%  ✓
chart_02      0.0004    100%  ✓
chart_03      0.0004    100%  ✓
textvqa_01    0.0004    100%  ✓
textvqa_02    0.0004    100%  ✓
textvqa_03    0.0004    100%  ✓
========================================
Avg loss: 0.0033
Avg overlap: 100.0%
```

**Key Findings:**
- Vision encoder collapse at 500 steps was due to insufficient training (resolved with more steps)
- SAM learns image-specific features with adequate training
- Two-stage pipeline successfully preserves DeepEncoder while training fresh GPT
- See [findings.md](../findings.md) for detailed analysis

### Tier 2 Data Strategy

**Goal:** Verify mixed data pipeline works and scales beyond Tier 1 overfitting.

**Datasets:**
- **FineVision** (HuggingFaceM4/FineVision) - 185 subsets, 24.3M samples (FineWeb for vision)
- **SFT data used in mid-training** - Karpathy's mid-training text recipe (reused from nanochat)

**Training Stages:**

| Stage | Data | Mix |
|-------|------|-----|
| Stage 1 | FineVision OCR subsets | 100% vision |
| Stage 2 | FineVision (70% OCR + 20% General) + Karpathy's mid traning text data (10% text) | 70/20/10 |

**Priority FineVision subsets:**
- DoclingMatix (1.27M) - PDF→Markdown
- SynthChartNet (500K) - Charts→OTSL

**Rationale:**
- FineVision provides unified conversation format (same as our Task pattern)
- Reuse Karpathy's battle-tested SFT dataset pipeline for text
- See [vision_tasks.md](vision_tasks.md) for full FineVision→DeepSeek-OCR mapping

## Implementation Notes (Tier-1)

### Code Refactoring
- Refactored to Karpathy style: config-at-top, generator functions, minimal abstractions
- `image_process.py`: 170 → 66 lines
- `vision_sample.py`: 411 → 148 lines
- Added `<|image|>` special token (vocab 65536 → 65537)

### Key Fixes Applied
- **Inference caching**: Vision embeddings cached during generation (2.5x speedup)
- **Image token masking**: Prevent model from generating `<|image|>` tokens
- **CLIP patch_embedding**: Marked as non-trainable (bypassed in our architecture)
- **DDP support**: `find_unused_parameters` for mixed vision+text training

### Multi-GPU Training
- Uses `DistributedDataParallel` (not Karpathy's DistMuon/DistAdamW)
- Vision encoder convolutions incompatible with Karpathy's custom optimizers
- `DistributedSampler` for data sharding across GPUs

## Related Docs

- [vision_tasks.md](vision_tasks.md) - Unified multimodal pipeline (tokenizer, dataloader, Task patterns)
- [DeepEncoder_loading_plan.md](DeepEncoder_loading_plan.md) - HuggingFace weight mappings
- [../findings.md](../findings.md) - Detailed experimental findings and debugging notes
- [../decisions.md](../decisions.md) - Technical decisions and rationale
