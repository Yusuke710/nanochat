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

1. **No special tokenizer methods** - just encode + expand
2. **Logic in forward()** - vision embedding merge is explicit
3. **One unified path** - same `tokenize_sample()` for vision and text
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
├── vision_dataloader.py      # Vision data generator
├── tokenizer.py              # Extended with <|image|> token
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
| 1 | `vis_tok_train.py` | SAM, CLIP, Projector, GPT | Nothing | Vision only | 4096 |
| 2 | `vis_mid_train.py` | CLIP, Projector, GPT | SAM + Conv | 90% vision + 10% text | 8192 |

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
| 1 | Overfit to near-zero loss on 10 images. 100% accuracy on `vision_sample.py`. | ✅ Complete |
| 2 | Mixed vision+text training on scaled data (details below) | Pending |
| 3 | Competitive scores on Fox/OmniDocBench | Pending |

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

- [vision_tasks.md](vision_tasks.md) - Vision dataloader design and Task patterns
- [DeepEncoder_loading_plan.md](DeepEncoder_loading_plan.md) - HuggingFace weight mappings
