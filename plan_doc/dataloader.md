# Vision Dataloader

Karpathy-style vision dataloader extending nanochat's Task pattern.

## Resolution Mode: Base Only (No Gundam)

| Mode | Resolution | Tokens | Process |
|------|------------|--------|---------|
| Tiny | 512 | 64 | resize |
| Small | 640 | 100 | resize |
| **Base** | **1024** | **256** | **padding** |
| Large | 1280 | 400 | padding |
| Gundam | 640+1024 | n×100+256 | crops+global |

**Current:** Base mode only. All images padded to 1024×1024, fixed 273 tokens per image.

## Design Principles

| Principle | How It's Applied |
|-----------|------------------|
| No special methods | Just `encode()` + expand tokens in a loop |
| Logic is explicit | Token expansion, masking all visible in `process_sample()` |
| One unified path | Same generator handles vision and text |
| Extends nanochat | Uses `Task`, `TaskMixture`, `collate` pattern |

## Why Discrete Batching (Not Token Packing)

nanochat has two patterns:

1. **Base pretraining** (`dataloader.py`): Token packing into B×T chunks
2. **SFT training** (`chat_sft.py`): Discrete batching with pad + collate

For vision, we use **SFT pattern** because:
- Each sample needs its own image-token alignment
- Can't pack tokens across images (breaks `<image>` position mapping)
- Fixed 273 tokens per image in Base mode

## Data Flow

```
dataset[i]
    ↓
{"text": "<image>\nOCR this.", "response": "Hello", "image_path": "doc.png"}
    ↓
process_sample()
    ├── tokenizer.encode() → [BOS, IMG, ..., text_tokens..., EOS]
    ├── expand <image> → [BOS, IMG, IMG, IMG, ... (×273), ..., EOS]
    └── process_image() → [3, 1024, 1024]
    ↓
collate()
    ├── pad to max_len
    ├── inputs = ids[:-1]
    ├── targets = ids[1:] with image positions = -1
    └── stack pixel_values
    ↓
model.forward(inputs, targets, pixel_values)
```

## Token Masking Strategy

**Pre-training loss**: Supervise ALL text tokens, mask only image positions.

```
# Result for "<image>\nOCR this." with 273 image tokens:
# inputs:  [BOS, IMG, IMG, ...(×273), \n,  O,   C,   R, ...]
# targets: [-1,  -1,  -1,  ...(×273), \n,  O,   C,   R, ...]
```

## Implementation

See `nanochat/vision_dataloader.py` for the Karpathy-style generator implementation.

## Future: Gundam Mode

When Base mode works for Tier-2, add dynamic resolution. See DeepSeek-OCR paper Section 3.2.2.
