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

## Model Forward

```python
class NanoDeepseekOCR(nn.Module):
    def forward(self, input_ids, targets=None, pixel_values=None, ...):
        # 1. Get text embeddings
        text_embeds = self.gpt.transformer.wte(input_ids)

        # 2. If images, encode and replace <image> positions
        if pixel_values is not None:
            vision_embeds = self.vision_encoder(pixel_values, ...)
            image_mask = (input_ids == self.image_token_id)
            text_embeds[image_mask] = vision_embeds.flatten(0, 1)

        # 3. Forward through GPT
        return self.gpt(inputs_embeds=text_embeds, targets=targets)
```

## Tokenization

One function handles both vision and text:

```python
def tokenize_sample(text, images=None):
    ids = tokenizer.encode(text, prepend="<|bos|>")

    if images is None:
        return ids, None

    pixel_values = []
    for img in images:
        pixel_values.append(processor.process_image(img))
        n_tokens = processor.count_tokens(img)
        ids = expand_image_token(ids, n_tokens)

    return ids, torch.stack(pixel_values)


def expand_image_token(ids, n_tokens):
    """Expand single <image> token to N copies."""
    result = []
    for tok in ids:
        if tok == IMAGE_TOKEN_ID:
            result.extend([IMAGE_TOKEN_ID] * n_tokens)
        else:
            result.append(tok)
    return result
```

Token count depends on resolution (73-900+ tokens). Expansion happens at data loading time.

## GPT Modification

Add `inputs_embeds` parameter to nanochat's `gpt.py`:

```python
def forward(self, idx=None, inputs_embeds=None, targets=None, ...):
    if inputs_embeds is not None:
        x = inputs_embeds
    else:
        x = self.transformer.wte(idx)
    # ... rest unchanged
```

## Loading nanochat from HuggingFace

```python
tokenizer = AutoTokenizer.from_pretrained("nanochat-students/base-d20")
tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})

model = build_nano_deepseek_ocr(gpt_checkpoint="nanochat-students/base-d20")
resize_embeddings(model, len(tokenizer))
model.set_image_token_id(tokenizer.convert_tokens_to_ids("<image>"))
```

## File Structure

```
nano-deepseek-ocr/
├── nano_deepseek_ocr.py      # Main VLM model
├── deepencoder/
│   ├── sam_vary_sdpa.py      # SAM encoder
│   ├── clip_sdpa.py          # CLIP encoder
│   └── load_pretrained.py    # HuggingFace weight loading
├── image_process.py          # Image → tensor
├── gpt.py                    # nanochat GPT (+ inputs_embeds)
├── tokenizer.py              # Add <image> token
└── scripts/
    ├── vis_tok_train.py      # Stage 1 training
    ├── vis_mid_train.py      # Stage 2 training
    └── vision_sample.py      # Test inference
```

## Training Stages

| Stage | Script | Trainable | Frozen | Data | Seq Len |
|-------|--------|-----------|--------|------|---------|
| 1 | `vis_tok_train.py` | SAM, CLIP, Projector, GPT | Nothing | Vision only | 4096 |
| 2 | `vis_mid_train.py` | CLIP, Projector, GPT | SAM + Conv | 90% vision + 10% text | 8192 |

### Optimizer Setup

```python
# Matrix params (Muon)
matrix_params = [*clip.parameters(), *projector.parameters(), *gpt.transformer.h.parameters()]
muon_optim = Muon(matrix_params, lr=0.02, momentum=0.95)

# Embedding params (AdamW)
embedding_params = [image_newline, view_separator, *gpt.transformer.wte.parameters(), gpt.lm_head.weight]
adam_optim = AdamW(embedding_params, lr=5e-5, betas=(0.9, 0.95), weight_decay=0.1)
```

### Stage Setup

```python
def setup_stage1(model):
    for p in model.parameters():
        p.requires_grad = True

def setup_stage2(model):
    for p in model.vision_encoder.sam.parameters():
        p.requires_grad = False
```

## Multi-Resolution Modes

| Mode | base_size | Tokens |
|------|-----------|--------|
| Tiny | 512 | 73 |
| Small | 640 | 111 |
| Base | 1024 | 273 |
| Large | 1280 | 421 |
| Gundam | 1024+crops | 273+ |

Token formula: `(num_queries + 1) × num_queries + 1` where `num_queries = resolution / patch_size / downsample_ratio`

## Implementation Order

1. Add `<image>` token to tokenizer
2. Add `inputs_embeds` to `gpt.py`
3. Create `nano_deepseek_ocr.py` with simple forward()
4. Create `image_process.py` for image → tensor
5. Create `load_pretrained.py` for HuggingFace weights
6. Create training scripts
7. Test on small data

## Evaluation

### Quick Testing

```bash
python -m scripts.vision_sample
python -m scripts.vision_sample --model_step 10000
```

Runs inference on 9 fixed test images (one per dataset), shows EXPECTED vs GENERATED.

### Success Criteria

| Tier | Criteria |
|------|----------|
| 1 | Overfit to near-zero loss on tiny dataset |
| 2 | Smooth training, basic OCR capability |
| 3 | Competitive scores on Fox/OmniDocBench |

## Related Docs

- [dataloader.md](dataloader.md) - Vision dataloader implementation
- [data_plan.md](data_plan.md) - Dataset details and Task class templates
- [DeepEncoder_loading_plan.md](DeepEncoder_loading_plan.md) - HuggingFace weight mappings
