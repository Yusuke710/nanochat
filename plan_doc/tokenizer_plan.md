# Tokenizer Plan for nano-deepseek-ocr

## Extending nanochat tokenizer for vision

### Existing nanochat special tokens:
```python
SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>", "<|user_end|>",
    "<|assistant_start|>", "<|assistant_end|>",
    "<|python_start|>", "<|python_end|>",
    "<|output_start|>", "<|output_end|>",
]
```

### New tokens needed for vision:

#### Required (core vision):
| Token | Purpose |
|-------|---------|
| `<image>` | Placeholder replaced with vision embeddings |

#### Optional (grounding/localization):
| Token | Purpose |
|-------|---------|
| `<\|grounding\|>` | Triggers grounding mode (outputs bboxes) |
| `<\|ref\|>` / `<\|/ref\|>` | Wrap text to locate in image |
| `<\|det\|>` / `<\|/det\|>` | Wrap detected object descriptions |

### Usage examples from DeepSeek-OCR:

```python
# Basic OCR
"<image>\n<|grounding|>Convert the document to markdown."

# Locate specific text
"<image>\nLocate <|ref|>xxxx<|/ref|> in the image."
```

### Extended SPECIAL_TOKENS:

```python
SPECIAL_TOKENS = [
    # existing
    "<|bos|>",
    "<|user_start|>", "<|user_end|>",
    "<|assistant_start|>", "<|assistant_end|>",
    "<|python_start|>", "<|python_end|>",
    "<|output_start|>", "<|output_end|>",
    # new for vision
    "<image>",           # REQUIRED - replaced with vision vectors
    # grounding (optional, for later)
    "<|grounding|>",
    "<|ref|>", "<|/ref|>",
    "<|det|>", "<|/det|>",
]
```

### Notes:
- The `<image>` token is special because its embedding is **never used** - it just marks positions where vision embeddings get inserted
- The grounding tokens are regular special tokens that the model learns to use for bbox output
- No padding token needed for images - padding is done in pixel space (gray pixels), not token space

---

## Vision Pre-training Tokenization {#vision-pre-training-tokenization}

For vision mid-training, we use a simple `render_vision_pretraining` method that handles `<image>` placeholders without conversation markers.

This is the **authoritative implementation** - referenced by [training_plan.md](training_plan.md) and [data_plan.md](data_plan.md).

### End Token Strategy

**Decision:** Use existing `<|assistant_end|>` as the stop token.

**Rationale:**
- nanochat has no `<|eos|>` token - only `<|bos|>` and turn markers
- Generation already stops on `<|assistant_end|>` (see `engine.py`)
- Reusing it keeps training/inference aligned with zero vocab changes
- Skipping an end token means the model won't learn when to stop

**Implementation:** Append `<|assistant_end|>` to the response text before tokenization:
```python
full_text = f"{prompt_text}{response_text}<|assistant_end|>"
```

### Signature

```python
def render_vision_pretraining(self, text, max_tokens=2048, image_token_counts=None):
    """
    Tokenize vision pre-training text with <image> placeholder expansion.

    Args:
        text: str - full text including prompt, response, and <|assistant_end|>
              e.g., "<image>\nOCR this.The document says...<|assistant_end|>"
        max_tokens: max sequence length
        image_token_counts: list of token counts per <image> tag

    Returns:
        ids: list[int] - token ids with <image> placeholders expanded
        image_positions: list[tuple] - (start, end) indices for each image

    Note: The caller is responsible for appending <|assistant_end|> to the text.
    BOS is automatically prepended by this method.
    """
```

### Implementation

```python
def render_vision_pretraining(self, text, max_tokens=2048, image_token_counts=None):
    ids = [self.bos_token_id]
    image_positions = []
    image_idx = 0

    # Split by <image> placeholder
    parts = text.split("<image>")
    for i, part in enumerate(parts):
        if part:
            part_ids = self.encode(part)
            ids.extend(part_ids)

        if i < len(parts) - 1:  # Not the last part = there's an <image> after
            # Expand <image> to N placeholder tokens
            n_tokens = image_token_counts[image_idx] if image_token_counts else 1
            image_start = len(ids)
            ids.extend([self.image_token_id] * n_tokens)
            image_positions.append((image_start, len(ids)))
            image_idx += 1

    # Truncate if needed
    if len(ids) > max_tokens:
        ids = ids[:max_tokens]
        # Filter image_positions to only include those within bounds
        image_positions = [(s, e) for s, e in image_positions if s < max_tokens]

    return ids, image_positions
```

### Usage Example

```python
# Build full text with end token
prompt = "<image>\nConvert this document to markdown."
response = "# Title\n\nThis is the document content..."
full_text = f"{prompt}{response}<|assistant_end|>"

# Tokenize
num_tokens = processor.process_images([image])["num_tokens_per_image"]
ids, img_pos = tokenizer.render_vision_pretraining(full_text, image_token_counts=num_tokens)

# Result: <|bos|>[273 image tokens]\nConvert this...content...<|assistant_end|>
```
