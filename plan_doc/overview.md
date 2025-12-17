# nano-deepseek-ocr: Implementation Overview

## Goal

Add vision capability to nanochat by replicating DeepSeek-OCR's training pipeline.

## Principles

1. **nanochat-compatible** — Same data format, same `TaskMixture` pattern
2. **Minimal changes** — Only adjust what's needed for nanochat
3. **Reference-faithful** — Follow `reference_code/DeepSeek-OCR/`

**Target**: ~8K lines, single GPU → 8xH100 scalable, reproducible via `speedrun.sh`

---

## Repository Structure

```
nano-deepseek-ocr/                 # forked from nanochat(the name could be nanochat)
├── nano_deepseek_ocr.py           # Main VLM model (DeepEncoder + nanochat GPT)
├── deepencoder/
│   ├── __init__.py
│   ├── sam_vary_sdpa.py           # SAM-ViT-B encoder (from DeepSeek-OCR)
│   ├── clip_sdpa.py               # CLIP-L/14 encoder (from DeepSeek-OCR)
│   ├── build_linear.py            # MLP projector (from DeepSeek-OCR)
│   └── load_pretrained.py         # HuggingFace weight loading
├── image_process.py               # DeepseekOCRProcessor (adapted, configurable prompt)
├── config.py                      # Configuration constants
├── gpt.py                         # nanochat GPT (reuse as-is)
├── tokenizer.py                   # Extended with <image> token
├── dataloader.py                  # Vision-aware data loading
├── dataset.py                     # Base dataset utilities
├── muon.py                        # Muon optimizer (from nanochat)
├── adamw.py                       # AdamW optimizer (from nanochat)
├── scripts/
│   ├── vis_tok_train.py           # Stage 1: Train ALL (DeepEncoder + nanochat GPT)
│   ├── vis_mid_train.py           # Stage 2: Train CLIP + LLM (freeze SAM + Conv)
│   └── vision_sample.py           # Manual sanity check: inference on fixed test images
├── data/
│   └── test_images/               # 9 fixed test images (one per dataset) with expected outputs
│       ├── expected.json          # Expected outputs for comparison
│       └── *.jpg, *.png           # Test images
├── tasks/
│   ├── common.py                  # VisionTaskMixture base class
│   ├─ other dataset
├── speedrun.sh                    # Full training pipeline
└── reference_code/
    └── DeepSeek-OCR/              # Original implementation (for guidance)
```

---

## Model Architecture {#model-architecture}

### DeepEncoder (Vision Encoder)

```
Image (H×W×3)
    │
    ▼
┌─────────────────────────────────────────┐
│ SAM-ViT-B (~92M params)                 │
│ - 768 embed, 12 layers, 12 heads        │
│ - Window attention + global at 2,5,8,11 │
│ - Output: (H/16, W/16, 768)             │
│ - Neck: Conv → LayerNorm → 256 channels │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Convolutional Compressor (net_2, net_3) │
│ - Conv2d: 256 → 512 → 1024              │
│ - 4× spatial reduction (16× token)      │
│ - Output: (H/64, W/64, 1024)            │
│ - RANDOM INIT (not in original SAM)     │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ CLIP-L/14 (~300M params)                │
│ - Bypasses patch_embedding layer        │
│ - Takes compressed SAM features directly│
│ - 24 layers, 1024 hidden dim            │
│ - Output: (N+1, 1024) with CLS token    │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Feature Concatenation + MLP Projector   │
│ - CLIP[:, 1:] || SAM.flat → 2048 dim    │
│ - Linear: 2048 → n_embd (768)           │
└─────────────────────────────────────────┘
    │
    ▼
Vision Tokens → merge at <image> positions → nanochat GPT
```

### Full Model (~1B params)

| Component | Params | Notes |
|-----------|--------|-------|
| SAM-ViT-B | ~92M | 86M from HuggingFace + 6M compressor (random init) |
| CLIP-L/14 | ~300M | From HuggingFace, patch_embedding bypassed |
| MLP Projector | ~1.5M | 2048 → n_embd |
| nanochat GPT | ~570M | Dense decoder |
| **Total** | **~970M** | All dense |

---

## Pretrained Weight Loading

Load SAM and CLIP from HuggingFace. See **[DeepEncoder_loading_plan.md](DeepEncoder_loading_plan.md)** for:
- Weight name mappings (HuggingFace → DeepSeek format)
- QKV fusion for CLIP
- Implementation code for `load_pretrained.py`

---

## Tokenizer Extension

Extend nanochat's tokenizer with vision tokens. See **[tokenizer_plan.md](tokenizer_plan.md)** for:
- New special tokens (`<image>`, `<|grounding|>`, etc.)
- `render_vision_pretraining()` implementation

---

## Training Stages

(see [training_plan.md](training_plan.md) for details)

### Stage 1: Joint Training (All Trainable) — `vis_tok_train.py`

| Component | Status |
|-----------|--------|
| SAM-ViT-B | **Train** |
| Conv Compressor | **Train** |
| CLIP-L | **Train** |
| MLP Projector | **Train** |
| nanochat GPT | **Train** |

**Data**: Vision-only, seq_len=4096, cosine scheduler

### Stage 2: Continued Training (Freeze SAM + Conv) — `vis_mid_train.py`

| Component | Status |
|-----------|--------|
| SAM-ViT-B | **FROZEN** |
| Conv Compressor | **FROZEN** |
| CLIP-L | **Train** |
| MLP Projector | **Train** |
| nanochat GPT | **Train** |

**Data**: Vision (90%) + Text-only (10%), seq_len=8192, step scheduler

---

## Data Formats

See **[data_plan.md](data_plan.md)** for complete dataset details (9 datasets, transformations, task classes).

**Vision Task Format**: `{text: "<image>\n...", response: "...", image_path: str}`

**Text Task Format**: nanochat conversation format `{messages: [...]}`

---

## Training Flow

```python
# Forward pass (vision-enabled)
vision_embeds = deep_encoder(pixel_values)      # (B, N, n_embd)
text_embeds = gpt.transformer.wte(input_ids)    # (B, T, n_embd)
inputs_embeds = merge_at_image_positions(input_ids, text_embeds, vision_embeds)
logits = gpt(inputs_embeds=inputs_embeds)
loss = F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=-1)
```

Same loss function as nanochat — only input processing changes.

---

## Evaluation

### Quick Visual Testing

Run `scripts/vision_sample.py` to test model quality on 9 fixed images:

```bash
python -m scripts.vision_sample --model_step 10000
```

Output shows EXPECTED vs GENERATED for each test case, enabling quick sanity checks without running full benchmarks. See [vision_sample.md](vision_sample.md) for details.

### Benchmarks (from DeepSeek-OCR)

- **Fox Benchmark**: 9 OCR tasks on dense PDF documents
- **OmniDocBench**: 1,355 PDF pages with rich annotations

### Success Criteria

| Tier | Criteria |
|------|----------|
| Tier 1 | Overfit to near-zero loss on tiny dataset |
| Tier 2 | Smooth training, basic OCR capability |
| Tier 3 | Competitive scores on Fox/OmniDocBench |

---

## Key Differences from DeepSeek-OCR

| Aspect | DeepSeek-OCR | nano-deepseek-ocr |
|--------|--------------|-------------------|
| LLM | DeepSeek V2/V3 (MoE) | nanochat (dense) |
| Total params | ~3B+ | ~1B |
| Training scale | 20 nodes × 8xA100 | 1 GPU → 8xH100 |
| Attention | Flash Attention | F.scaled_dot_product_attention |
| Framework | vLLM-integrated | Standalone PyTorch |
| Complexity | Production-grade | Educational/hackable |

---

## Related Planning Documents

- [nano_deepseek_ocr_blueprint.md](nano_deepseek_ocr_blueprint.md) — Main model code blueprint
- [DeepEncoder_loading_plan.md](DeepEncoder_loading_plan.md) — HuggingFace weight loading
- [tokenizer_plan.md](tokenizer_plan.md) — Tokenizer extension
- [training_plan.md](training_plan.md) — Training pipeline details
- [vision_sample.md](vision_sample.md) — Manual testing script for visual inspection
