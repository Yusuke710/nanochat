# Gundam Mode Implementation Plan

Multi-resolution dynamic crop processing for high-resolution images (DeepSeek-OCR Section 3.2.2).

> **Note**: Code snippets below are **references to understand the algorithm**. Make sure your actual implementation is **dead simple** - Karpathy style with minimal abstractions, clear data flow, and no hidden magic.

## Overview

**Current (Base mode)**:
- Single 1024×1024 image
- 273 tokens per image
- Loses detail in high-res documents

**Gundam mode**:
- 1 **global view**: 1024×1024 → 273 tokens (with newlines + separator)
- N **local crops**: 640×640 each → 100 tokens per crop (10×10 grid)
- Token order: `[LOCAL_CROPS, GLOBAL_VIEW, VIEW_SEPARATOR]`

## Token Calculation

From reference `image_process.py:424-435`:

```python
# Global view (base_size=1024)
num_queries_base = 1024 // 16 // 4 = 16
global_tokens = (16 + 1) * 16 + 1 = 273  # includes newlines + separator

# Local crops (image_size=640)
num_queries = 640 // 16 // 4 = 10
local_tokens = (10 * w_tiles + 1) * (10 * h_tiles)  # includes newlines
```

| Crop Grid | Local Tokens | Total Tokens |
|-----------|--------------|--------------|
| 1×1 (no crops) | 0 | 273 |
| 2×1 | 210 | 483 |
| 1×2 | 220 | 493 |
| 2×2 | 440 | 713 |
| 3×2 | 660 | 933 |

## Reference Code Analysis

### Key Files
- `reference/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py` - Config params
- `reference/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/image_process.py` - Crop logic
- `reference/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py` - Vision encoding

### Config Parameters (config.py:1-15)
```python
# Gundam mode settings
BASE_SIZE = 1024     # Global view resolution
IMAGE_SIZE = 640     # Local crop resolution
CROP_MODE = True     # Enable cropping
MIN_CROPS = 2        # Minimum crops
MAX_CROPS = 6        # Maximum crops (9 max, 6 recommended for memory)
```

### Crop Selection Algorithm (image_process.py:11-43)

The algorithm finds the best (w_tiles, h_tiles) grid based on image aspect ratio:

1. Generate all valid tile combinations: `{(1,2), (2,1), (2,2), (1,3), ...}` where `min_crops <= w*h <= max_crops`
2. Find the combination whose aspect ratio is closest to the image's aspect ratio
3. Resize image to `(640 * w_tiles, 640 * h_tiles)`
4. Crop into `w_tiles * h_tiles` blocks of 640×640

### Vision Encoding (deepseek_ocr.py:363-467)

**Key insight**: Local crops come BEFORE global view in token sequence.

1. Encode local crops (640×640 each) → SAM + CLIP + projector → 100 tokens each
2. Encode global view (1024×1024) → SAM + CLIP + projector → 256 tokens
3. Reshape and add newline tokens at end of each row
4. Concatenate: `[local_flat, global_flat, view_separator]`

## Implementation Steps

### Phase 1: Data Preparation

**Create `vlm-gundam10` dataset**:

1. Collect 10 high-resolution images (>2000px on longest edge):
   - 4 high-res receipts (small text, dense layout)
   - 3 high-res charts (fine axis labels, legends)
   - 3 high-res scene text (billboards, signs)

2. Use same format as `vlm-overfit10`:
```json
{
  "id": "gundam_receipt_000",
  "image": "images/gundam_receipt_000.jpg",
  "source": "SROIE_highres",
  "type": "receipt_ocr",
  "prompt": "<image>\n<|grounding|>OCR this image.",
  "answer": "..."
}
```

3. Create upload script `scripts/upload_vlm_gundam10.py` (copy from `upload_vlm_overfit10.py`)

### Phase 2: Image Processing

**Modify `nanochat/image_process.py`**:

Add simple functions for:
- `find_closest_aspect_ratio()` - Find best crop grid
- `dynamic_preprocess()` - Split image into crops
- `process_image_gundam()` - Returns global view + local crops
- `count_gundam_tokens()` - Calculate token count for a given crop grid

### Phase 3: Dataloader

**Modify `nanochat/multimodal_dataloader.py`**:

- Add `gundam_mode` flag to `MultimodalDataset`
- Handle variable token counts per image
- Return crop metadata in batch

### Phase 4: Vision Encoding

**Modify `nanochat/nano_deepseek_ocr.py`**:

- Add `encode_images_gundam()` method
- Process local crops and global view separately
- Add newlines and combine with correct token order

### Phase 5: Training Scripts

**Add Gundam config to `scripts/vis_tok_train.py`**:

- Add `gundam_mode` flag
- Conditional forward pass for Gundam vs Base mode

## Memory Considerations

| Mode | Max Tokens | Approx VRAM (B=1, seq=4096) |
|------|------------|------------------------------|
| Base | 273 | ~8GB |
| Gundam 2×1 | 483 | ~12GB |
| Gundam 2×2 | 713 | ~16GB |
| Gundam 3×2 | 933 | ~22GB |

**Recommendations**:
- Start with `max_crops=2` for testing
- Use gradient checkpointing for larger crop counts
- Consider batch_size=1 for Gundam mode on smaller GPUs

## Testing Strategy

1. **Unit tests**:
   - `count_tiles()` returns correct grid for various aspect ratios
   - `dynamic_preprocess()` produces correct number of crops
   - Token count matches reference implementation

2. **Integration test**:
   - Forward pass works with Gundam tensors
   - Loss decreases during training

3. **Overfitting test**:
   - Train on `vlm-gundam10` with high-res images
   - Verify model learns fine-grained details

## Files to Modify

```
nanochat/
├── image_process.py           # Add crop functions
├── multimodal_dataloader.py   # Handle variable tokens
├── nano_deepseek_ocr.py       # Add encode_images_gundam()
scripts/
├── vis_tok_train.py           # Add gundam_mode config
├── upload_vlm_gundam10.py     # New: upload script
tasks/
├── vlm_gundam10.py            # New: dataset loader
```

## References

- [DeepSeek-OCR Paper (arXiv:2510.18234)](https://arxiv.org/html/2510.18234v1) - Section 3.2.2
- [DeepWiki: Image Processing Pipeline](https://deepwiki.com/deepseek-ai/DeepSeek-OCR/4.4-image-processing-pipeline)
- Reference code: `reference/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/`
