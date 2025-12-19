# nano-deepseek-ocr Plan

Add vision capability to nanochat using DeepSeek-OCR's vision encoder.

## Architecture

```
Image (H×W×3)
    │
    ▼
┌─────────────────────────────────────────┐
│ SAM-ViT-B (~92M params)                 │
│ - 768 embed, 12 layers, 12 heads        │
│ - Output: (H/16, W/16, 768)             │
│ - Conv compressor: 4× reduction         │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ CLIP-L/14 (~300M params)                │
│ - Takes compressed SAM features         │
│ - 24 layers, 1024 hidden dim            │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ MLP Projector (~1.5M)                   │
│ - CLIP + SAM features → n_embd          │
└─────────────────────────────────────────┘
    │
    ▼
Vision Embeddings → Replace <image> positions → nanochat GPT (~570M)
```

**Total: ~970M params**

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
| 1 | `vis_tok_train.py` | SAM, CLIP, Projector, GPT | Nothing | `TaskMixture([VisionTask()])` | 4096 |
| 2 | `vis_mid_train.py` | CLIP, Projector, GPT | SAM + Conv | `TaskMixture([VisionTask(), TextTask()])` | 8192 |

Both stages use `multimodal_data_generator()` - only the TaskMixture contents change.

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
python -m scripts.vision_sample
python -m scripts.vision_sample --resume_step=150
```

### Success Criteria

| Tier | Criteria | Status |
|------|----------|--------|
| 1 | Overfit to near-zero loss on 10 images. 100% accuracy on `vision_sample.py`. | 🔄 Re-run with unified pipeline |
| 2 | Mixed vision+text training on scaled data (details below) | Pending |
| 3 | Competitive scores on Fox/OmniDocBench | Pending |

### Tier 1 Re-validation (Unified Pipeline)

**Why re-run:** New `multimodal_dataloader.py` and `MEDIA_PLACEHOLDERS` in tokenizer. Must verify the unified pipeline works before scaling.

**Data:**

| Stage | TaskMixture | Purpose |
|-------|-------------|---------|
| Stage 1 | `OverfitSamples(data_dir="data")` | Vision-only, 10 images |
| Stage 2 | `OverfitSamples(...) + SmolTalk(limit=10)` | Mixed training validation |

**Success criteria:**
- Stage 1: Near-zero loss, 100% accuracy on `vision_sample.py`
- Stage 2: Loss converges on both vision and text samples

### Tier 2 Data Strategy

**Goal:** Verify mixed data pipeline works and scales beyond Tier 1 overfitting.

**Datasets (1k samples each):**
- **olmOCR** (allenai/olmOCR-mix-1025) - OCR/document understanding
- **LLaVA-CC3M-Pretrain-595K** - image captioning/general vision
- **SmolTalk** - text-only conversation data

**Training Stages:**

| Stage | Data | Purpose |
|-------|------|---------|
| Stage 1 | olmOCR (1k) + LLaVA-CC3M-Pretrain-595K (1k) | Vision-only pretraining |
| Stage 2 | olmOCR (1k) + LLaVA-CC3M-Pretrain-595K (1k) + SmolTalk (1k) | Mixed vision+text fine-tuning |

**Rationale:**
- Small scale (3k total) enables quick iteration
- Two vision datasets ensure diversity in vision tasks
- Text data in Stage 2 validates mixed modality training
- Confirms data pipeline scales before moving to Tier 3 full datasets

## Related Docs

- [vision_tasks.md](vision_tasks.md) - Unified multimodal pipeline (tokenizer, dataloader, Task patterns)
- [DeepEncoder_loading_plan.md](DeepEncoder_loading_plan.md) - HuggingFace weight mappings
