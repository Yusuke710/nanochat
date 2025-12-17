# Training Plan: Reusing nanochat's Pipeline for DeepSeek-OCR

## Overview

**Key Insight**: Both nanochat and DeepSeek-OCR use identical next-token prediction loss. The training loop (optimizer, scheduler, checkpointing) can be largely reused, but the **dataloader must be redesigned** from streaming token packing to discrete sample batching to preserve vision-text alignment. See Section 9 for details.

---

## 1. Why Reuse Works

### Same Loss Function

```python
# Both nanochat and DeepSeek-OCR use pre-training style loss:
loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    targets.view(-1),
    ignore_index=-1  # Only mask image placeholder positions
)
```

### Same Training Objective: Mid-Training (Not SFT)

Vision mid-training follows nanochat's `mid_train.py` pattern - **supervise ALL text tokens**, not selective supervision:

| Aspect | nanochat `base_train` | nanochat `mid_train` | Vision Mid-Training |
|--------|----------------------|---------------------|---------------------|
| Objective | Next-token prediction | Next-token prediction | Next-token prediction |
| Supervision | All tokens | All tokens (ignore mask) | All text tokens |
| Masking | None | None (mask ignored) | Only image placeholders |
| Optimizer | Muon + AdamW | Muon + AdamW | Muon + AdamW |
| Data format | Raw text | Conversations | `<image>\nprompt{response}` (plain) |

**Key insight**: Vision training uses **plain prompts** (no conversation markers), matching DeepSeek-OCR paper format:
```python
# Paper uses prompts like: "<image>\nFree OCR." or "<image>\n<|grounding|>Convert the document to markdown."
# We append <|assistant_end|> to teach the model when to stop (consistent with nanochat inference)
full_text = f"{text}{response}<|assistant_end|>"
ids, img_pos = tokenizer.render_vision_pretraining(full_text, image_token_counts)
```

This is simpler than nanochat's conversation format and matches the actual DeepSeek-OCR training.

---

## 2. DeepseekOCRProcessor is Standalone

The processor is **decoupled from the vision encoder** and runs during **data loading (CPU)**, not during the forward pass (GPU):

```python
# Can use independently without loading SAM/CLIP
from nano_dpsk_ocr.image_process import DeepseekOCRProcessor

processor = DeepseekOCRProcessor(tokenizer, image_size=640, ...)

# Produces all tensors needed for training
output = processor.tokenize_with_images(
    images=[pil_image],
    text="<image>\nOCR this document.",
    ...
)
# Returns: input_ids, pixel_values, images_seq_mask, etc.
```

**Note**: We use `pixel_values`, `images_spatial_crop`, and image token counts from the processor output. Tokenization uses `render_vision_pretraining()` to handle `<image>` placeholders - see Section 5.1.

---

## 3. TaskMixture Reusability

### nanochat's TaskMixture Works As-Is

```python
class TaskMixture(Task):
    """Mix multiple datasets with deterministic shuffling."""

    def __init__(self, tasks):
        self.tasks = tasks
        self.index_map = [...]  # (task_idx, local_idx) pairs
        random.Random(42).shuffle(self.index_map)

    def get_example(self, index):
        task_idx, local_idx = self.index_map[index]
        return self.tasks[task_idx][local_idx]
```

`TaskMixture` is agnostic to what each task returns - it just calls `get_example()`. **Mixed formats are supported.**

**Important caveats**:
1. **Format normalization required**: nanochat text tasks (SmolTalk, GSM8K) return `{messages: [...]}` while vision tasks return `{text, response, image_path}`. The data generator must normalize all samples via `normalize_sample()` before processing. See "Sample Format Adapter" section below.
2. **Discrete batching required**: nanochat's `mid_data_generator` uses streaming token packing that destroys sample boundaries. For vision training, we must replace the data generator with discrete sample batching. See **Section 9** for the solution.

### Unified Plain Prompt Format

**Key Design Decision**: ALL vision data uses **plain prompts** (no conversation markers), matching DeepSeek-OCR paper's actual training format.

```python
# Format 1: Vision data (ALL types - OCR, documents, charts, LLaVA, etc.)
# Uses render_vision_pretraining() - NO conversation markers
{
    "text": "<image>\nFree OCR.",           # prompt
    "response": "The document says...",      # model output
    "image_path": str
}
# Tokenizes to: <|bos|>[image_tokens × N]\nFree OCR.The document says...<|assistant_end|>
# The <|assistant_end|> is appended during data loading (teaches model when to stop)
# Loss: supervise ALL text tokens (pre-training style)

# LLaVA-style data converted to plain prompt:
{
    "text": "<image>\nDescribe this image in detail.",
    "response": "A cat sitting on a couch...",
    "image_path": str
}
# Same tokenization - NO conversation markers

# Format 2: Text-only data (Stage 2 only, preserves LM ability)
{
    "text": "What is 2+2?",
    "response": "4"
    # No image_path field
}
# Tokenizes to: <|bos|>What is 2+2?4<|assistant_end|>
# Loss: supervise ALL tokens
```

**Important**: The `<image>` placeholder expands to **64-1000+ tokens** based on resolution mode. See Section 6 for details.

### Sample Format Adapter (Compatibility Layer)

**Problem**: nanochat text tasks (SmolTalk, GSM8K, etc.) return `{messages: [...]}` format, but our vision pipeline expects `{text, response}`. These formats are incompatible - passing `{messages}` to the data generator will crash when accessing `sample['text']`.

**Solution**: Normalize ALL samples to `{text, response}` format at the data source boundary, before the data generator sees them.

```python
# nano_dpsk_ocr/data/sample_adapter.py

def normalize_sample(sample: dict) -> dict:
    """
    Normalize any sample format to {text, response, ...} schema.

    Accepts:
      - {text, response, [image_path]} - vision format (passthrough)
      - {messages: [{role, content}, ...]} - nanochat conversation format

    Returns:
      - {text, response, [image_path], [_source_format]} - normalized format

    Raises:
      - ValueError if sample has neither valid format
    """
    has_text_response = "text" in sample and "response" in sample
    has_messages = "messages" in sample

    # Validation: exactly one format must be present
    if has_text_response and has_messages:
        raise ValueError(
            f"Ambiguous sample format: has both 'text'/'response' AND 'messages'. "
            f"Keys: {list(sample.keys())}"
        )
    if not has_text_response and not has_messages:
        raise ValueError(
            f"Invalid sample format: missing both 'text'/'response' and 'messages'. "
            f"Keys: {list(sample.keys())}"
        )

    # Passthrough: already in {text, response} format
    if has_text_response:
        return sample

    # Convert: {messages} -> {text, response}
    messages = sample["messages"]

    # Handle optional system message (merge into first user message)
    if messages[0]["role"] == "system":
        system_content = messages[0]["content"]
        messages = messages[1:]
        if messages and messages[0]["role"] == "user":
            messages[0] = {
                "role": "user",
                "content": f"{system_content}\n\n{messages[0]['content']}"
            }

    # Extract user turns as "text", assistant turns as "response"
    # For multi-turn: concatenate all turns with role markers
    if len(messages) == 2 and messages[0]["role"] == "user" and messages[1]["role"] == "assistant":
        # Simple single-turn: direct mapping
        text = messages[0]["content"]
        response = messages[1]["content"]
    else:
        # Multi-turn: flatten with minimal markers
        # This preserves the conversation but fits our {text, response} schema
        text_parts = []
        response_parts = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, list):
                # Handle GSM8K-style list-of-parts content
                content = "".join(
                    p["text"] if isinstance(p, dict) and "text" in p else str(p)
                    for p in content
                )
            if msg["role"] == "user":
                text_parts.append(content)
            else:  # assistant
                response_parts.append(content)

        # Join with newlines (simple, no special tokens)
        text = "\n".join(text_parts)
        response = "\n".join(response_parts)

    # Build normalized sample, preserving any extra metadata
    normalized = {
        "text": text,
        "response": response,
        "_source_format": "messages",  # Track original format for debugging
    }

    # Copy through any other fields (e.g., image_path, subject, letters)
    for key, value in sample.items():
        if key not in ("messages", "text", "response"):
            normalized[key] = value

    return normalized


def validate_sample(sample: dict) -> None:
    """
    Fail-fast validation. Call after normalize_sample() to catch issues early.

    Raises ValueError with clear message if sample is malformed.
    """
    if "text" not in sample:
        raise ValueError(f"Sample missing 'text' field. Keys: {list(sample.keys())}")
    if "response" not in sample:
        raise ValueError(f"Sample missing 'response' field. Keys: {list(sample.keys())}")

    if not isinstance(sample["text"], str):
        raise ValueError(f"Sample 'text' must be str, got {type(sample['text'])}")
    if not isinstance(sample["response"], str):
        raise ValueError(f"Sample 'response' must be str, got {type(sample['response'])}")

    # Vision samples must have valid image_path
    if "image_path" in sample:
        if not isinstance(sample["image_path"], str):
            raise ValueError(f"Sample 'image_path' must be str, got {type(sample['image_path'])}")
```

**Why this works**:
- Tokenizer (`render_conversation`) stays untouched - we don't use it for vision training
- Single adapter handles all format conversion at the boundary
- Fail-fast validation catches issues before they reach the data generator
- `_source_format` metadata aids debugging without affecting training

### Data Generator Flow

```python
from nano_dpsk_ocr.data.sample_adapter import normalize_sample, validate_sample

sample = dataset[cursor]

# === ADAPTER: Normalize format at boundary ===
sample = normalize_sample(sample)  # {messages} -> {text, response}
validate_sample(sample)            # Fail fast on malformed data

# Build full text with <|assistant_end|> stop token
full_text = f"{sample['text']}{sample['response']}<|assistant_end|>"

if "image_path" in sample:
    # Vision data - plain prompt format
    image = Image.open(sample["image_path"])
    num_tokens = processor.count_image_tokens(image, mode=resolution_mode)
    ids, img_pos = tokenizer.render_vision_pretraining(
        full_text, image_token_counts=[num_tokens]
    )
    pixel_values = processor.preprocess_image(image, mode=resolution_mode)
else:
    # Text-only data
    ids = [bos_token_id] + tokenizer.encode(full_text)
    img_pos = []
    pixel_values = None

# Build targets: standard next-token prediction
targets = ids[1:]  # shift by 1
# Only mask image placeholder positions
for start, end in img_pos:
    targets[start:end-1] = -1  # ignore_index for image tokens
```

### Usage (same as nanochat)

```python
train_ds = TaskMixture([
    # Vision tasks (plain prompt format)
    DocLayNetOCR(split="train"),     # {text: "<image>\nFree OCR.", response: "...", image_path: ...}
    ChartQA(split="train"),          # {text: "<image>\nParse the figure.", response: "...", ...}
    LLaVADescribe(split="train"),    # {text: "<image>\nDescribe...", response: "...", ...}
    # Text-only tasks (Stage 2)
    SmolTalk(split="train"),
])
```

---

## 4. Training Stages

### Stage 1: Joint DeepEncoder + nanochat Training — `vis_tok_train.py`

**Following Paper Section 3.5.1**: Train DeepEncoder with LM decoder using next-token prediction. We use the same nanochat model for both stages (simpler than paper's compact LM approach).

**What's Trained:**
- SAM-ViT-B encoder (~80M)
- Convolutional Compressor
- CLIP-L encoder (~300M)
- MLP Projector
- Special embeddings (`image_newline`, `view_separator`)
- **nanochat GPT** - fully trainable (NOT frozen)

**What's Frozen:**
- Nothing - all components are trainable

**Data:**
- All vision data (OCR, documents, charts, LLaVA-style, etc.)
- Vision-only in this stage

**Training Config:**
- Sequence length: **4096 tokens**
- Batch size: 1280 (global, adjust for available GPUs)
- Epochs: 2
- Optimizer (Muon + AdamW):
  ```python
  # Matrix parameters (linear layers) - Muon
  # NOTE: Using Muon for vision encoder is experimental. If training is unstable
  # or diverges, lower the vision LR (e.g., 0.002 instead of 0.02) or switch
  # vision params to AdamW with lr=1e-4.
  matrix_params = [
      *sam_encoder.parameters(),
      *conv_compressor.parameters(),
      *clip_encoder.parameters(),
      *projector.parameters(),
      *gpt.transformer.h.parameters(),  # nanochat layers
  ]
  muon_optim = Muon(matrix_params, lr=0.02, momentum=0.95)

  # Embedding/scalar parameters - AdamW
  embedding_params = [
      image_newline, view_separator,
      *gpt.transformer.wte.parameters(),
      gpt.lm_head.weight,  # if not tied
  ]
  adam_optim = AdamW(embedding_params, lr=5e-5, betas=(0.9, 0.95), weight_decay=0.1)

  # Learning rate schedule: cosine annealing
  scheduler = CosineAnnealingLR(total_steps, warmup_steps=2000)
  ```

**Loss:**
- Same cross-entropy next-token prediction

---

### Stage 2: Continued Training (SAM+Compressor Frozen) — `vis_mid_train.py`

**Following Paper Section 3.5.2**: Freeze SAM+Compressor, continue training CLIP + nanochat.

**What's Trained:**
- CLIP-L encoder
- MLP Projector
- Special embeddings
- nanochat GPT

**What's Frozen:**
- SAM-ViT-B encoder (frozen after Stage 1)
- Convolutional Compressor (frozen after Stage 1)

**Rationale:**
- SAM already produces good local features after Stage 1
- Conv compressor is a simple spatial reducer, doesn't need more training
- Focus training budget on CLIP (global reasoning) and LLM

**Data:**
- Vision data (90%) + text-only data (10%)
- Text-only preserves nanochat's language capabilities

**Training Config:**
- Sequence length: **8192 tokens**
- Batch size: 640 (global, adjust for available GPUs)
- Optimizer (Muon + AdamW):
  ```python
  # Freeze SAM and Conv compressor
  for p in sam_encoder.parameters():
      p.requires_grad = False
  for p in conv_compressor.parameters():
      p.requires_grad = False

  # Matrix parameters (linear layers) - Muon
  # NOTE: If training diverges, lower CLIP LR (e.g., 0.002) or use AdamW for CLIP.
  matrix_params = [
      *clip_encoder.parameters(),
      *projector.parameters(),
      *gpt.transformer.h.parameters(),
  ]
  muon_optim = Muon(matrix_params, lr=0.02, momentum=0.95)

  # Embedding/scalar parameters - AdamW
  embedding_params = [
      image_newline, view_separator,
      *gpt.transformer.wte.parameters(),
      gpt.lm_head.weight,
  ]
  adam_optim = AdamW(embedding_params, lr=3e-5, betas=(0.9, 0.95), weight_decay=0.1)

  # Learning rate schedule: step-based (decay at milestones)
  scheduler = MultiStepLR(milestones=[...], gamma=0.1)
  ```

**Loss:**
- Same cross-entropy next-token prediction

---

## 5. Required Changes from nanochat

### 5.0 Modify nanochat GPT to Accept `inputs_embeds` (CRITICAL)

**Why needed**: Vision-language models merge vision embeddings with text embeddings BEFORE feeding to the transformer. nanochat's GPT currently only accepts `idx` (token IDs) and computes embeddings internally. We need to bypass this to inject pre-merged vision+text embeddings.

**Current nanochat GPT forward** ([gpt.py:243-274](../reference_code/nanochat/nanochat/gpt.py)):
```python
def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
    B, T = idx.size()
    # ...
    x = self.transformer.wte(idx)  # <-- Only accepts token IDs
    x = norm(x)
    for block in self.transformer.h:
        x = block(x, cos_sin, kv_cache)
    # ...
```

**Required modification** (add `inputs_embeds` parameter):
```python
def forward(self, idx=None, inputs_embeds=None, targets=None, kv_cache=None, loss_reduction='mean'):
    """
    Forward pass with optional pre-computed embeddings.

    Args:
        idx: Token IDs (B, T) - used if inputs_embeds is None
        inputs_embeds: Pre-computed embeddings (B, T, n_embd) - for VLM use
        targets: Target token IDs for loss computation
        kv_cache: KV cache for inference
        loss_reduction: 'mean' or 'none'

    Note: Either idx OR inputs_embeds must be provided, not both.
    """
    # Handle inputs_embeds vs idx
    if inputs_embeds is not None:
        B, T, _ = inputs_embeds.size()
        x = inputs_embeds
    else:
        assert idx is not None, "Either idx or inputs_embeds must be provided"
        B, T = idx.size()
        x = self.transformer.wte(idx)

    # Grab the rotary embeddings for the current sequence length
    assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
    assert x.device == self.cos.device, f"Rotary embeddings and input are on different devices"
    assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"

    # if kv cache exists, offset the rotary embeddings
    T0 = 0 if kv_cache is None else kv_cache.get_pos()
    cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

    # Forward the trunk of the Transformer
    x = norm(x)
    for block in self.transformer.h:
        x = block(x, cos_sin, kv_cache)
    x = norm(x)

    # Forward the lm_head (compute logits)
    softcap = 15
    logits = self.lm_head(x)
    logits = logits.float()
    logits = softcap * torch.tanh(logits / softcap)

    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
        return loss
    else:
        return logits
```

**Usage in NanoDeepseekOCR**:
```python
def forward(self, input_ids, targets=None, pixel_values=None, ...):
    # Get text embeddings
    text_embeds = self.gpt.transformer.wte(input_ids)

    # Encode and merge vision embeddings if images provided
    if pixel_values is not None:
        vision_embeds = self.vision_encoder.encode_image(pixel_values, ...)
        inputs_embeds = self.merge_vision_embeddings(input_ids, text_embeds, vision_embeds)
    else:
        inputs_embeds = text_embeds

    # Forward through GPT with pre-computed embeddings
    return self.gpt(inputs_embeds=inputs_embeds, targets=targets)
```

**Key points**:
1. **Backward compatible**: When `inputs_embeds=None`, behaves exactly like original nanochat
2. **No code duplication**: Avoids re-implementing GPT forward logic in NanoDeepseekOCR
3. **KV cache compatible**: Works with inference caching
4. **Minimal change**: Only ~10 lines modified in `gpt.py`

---

### 5.1 Tokenization Architecture: Plain Prompt Only

**Design Decision**: ALL vision data uses **plain prompts** (no conversation markers), matching DeepSeek-OCR paper format. This is simpler than nanochat's dual-path approach.

#### Single Tokenization Path

| Method | Use Case | Format | Loss |
|--------|----------|--------|------|
| `render_vision_pretraining()` | ALL vision data | `<\|bos\|><image>\nprompt{response}` | All text tokens |

#### Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         Vision Training Pipeline                           │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ALL Vision Data (OCR, LLaVA, Charts, etc.)                               │
│  ─────────────────────────────────────────                                │
│  render_vision_pretraining()                                               │
│  <|bos|><image>\nprompt{response}                                         │
│           │                                                                │
│           ▼                                                                │
│  VisionProcessor.process_images()                                          │
│  - Multi-resolution mode selection (Tiny/Small/Base/Large/Gundam)         │
│  - Returns: pixel_values, num_tokens_per_image                             │
│           │                                                                │
│           ▼                                                                │
│  collate_multimodal_batch()                                                │
│  - Pads sequences, stacks images                                           │
│  - Returns: inputs, targets, pixel_values                                  │
│           │                                                                │
│           ▼                                                                │
│  Pre-training Loss (supervise all text tokens)                             │
│  targets[image_positions] = -1  (mask only image tokens)                   │
│                                                                            │
└───────────────────────────────────────────────────────────────────────────┘
```

#### `render_vision_pretraining()` (NEW)

**See [tokenizer_plan.md](tokenizer_plan.md#vision-pre-training-tokenization)** for the full implementation.

Tokenizes vision pre-training samples: `<|bos|><image>\nprompt{response}` - NO conversation markers, matches DeepSeek-OCR paper format.

#### Component 2: `VisionProcessor.process_images`

Extracted from `tokenize_with_images()`, handles only image → tensor conversion:

```python
class VisionProcessor:
    def process_images(self, images: List[Image.Image]) -> dict:
        """
        Process images into tensors for the vision encoder.

        Returns:
            pixel_values: (N_images, 3, base_size, base_size) - global views
            images_crop: (N_images, N_crops, 3, image_size, image_size) - local crops
            images_spatial_crop: (N_images, 2) - [width_tiles, height_tiles]
            num_tokens_per_image: List[int] - how many tokens each image produces
        """
        # Dynamic tiling logic from tokenize_with_images
        # Transform and normalize images
        # Return tensors without any tokenization
```

#### Token Count Calculation Formula

**Critical**: The number of `<image>` tokens must match exactly what DeepEncoder produces. This formula is from [image_process.py](../reference_code/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/image_process.py):

```python
def count_image_tokens(
    image_width: int,
    image_height: int,
    image_size: int = 640,      # Local crop size
    base_size: int = 1024,      # Global view size
    patch_size: int = 16,       # ViT patch size
    downsample_ratio: int = 4,  # Conv compressor ratio (16× total: 4×4)
    cropping: bool = True,
) -> int:
    """
    Calculate the number of vision tokens for an image.

    Token layout (2D format with newlines):
    - Global view: h_base rows, each with (w_base + 1) tokens (w_base features + newline)
    - Local crops: h_local rows, each with (w_local + 1) tokens (if crops exist)
    - Final: +1 for view_separator

    Returns:
        Total number of image tokens
    """
    # Compute spatial dimensions after patch + downsample
    # For base_size=1024: 1024/16/4 = 16
    # For image_size=640: 640/16/4 = 10
    num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)  # 16 for 1024
    num_queries = math.ceil((image_size // patch_size) / downsample_ratio)       # 10 for 640

    # Global view tokens: (h + 1 newline) × h + 1 view_separator
    # Pattern: [row0_features + newline][row1_features + newline]...[view_sep]
    # = (num_queries_base + 1) × num_queries_base + 1
    global_tokens = (num_queries_base + 1) * num_queries_base + 1

    # Determine crop grid (only if image > 640×640)
    if cropping and (image_width > 640 or image_height > 640):
        num_width_tiles, num_height_tiles = count_tiles(image_width, image_height, image_size)
    else:
        num_width_tiles, num_height_tiles = 1, 1

    # Local crop tokens (only if multiple tiles)
    if num_width_tiles > 1 or num_height_tiles > 1:
        # Pattern: for each row of tiles, output (w_tiles × w_queries) features + 1 newline
        # Rows = h_tiles × h_queries
        # local_tokens = (h_tiles × num_queries) × (w_tiles × num_queries + 1)
        local_tokens = (num_height_tiles * num_queries) * (num_width_tiles * num_queries + 1)
    else:
        local_tokens = 0

    return global_tokens + local_tokens


# Examples (token counts include newline + view_separator):
# Fixed-resolution modes (crop_mode=False):
# - Tiny:  base_size=512,  num_queries=8,  global=(8+1)×8+1=73,   total=73
# - Small: base_size=640,  num_queries=10, global=(10+1)×10+1=111, total=111
# - Base:  base_size=1024, num_queries=16, global=(16+1)×16+1=273, total=273
# - Large: base_size=1280, num_queries=20, global=(20+1)×20+1=421, total=421
#
# Gundam mode (crop_mode=True, base_size=1024 global + 640 local tiles):
# - 800×600 image (2×1 tiles):  global=273, local=(1×10)×(2×10+1)=210, total=483
# - 1280×640 image (2×1 tiles): global=273, local=(1×10)×(2×10+1)=210, total=483
# - 1920×1280 image (3×2 tiles): global=273, local=(2×10)×(3×10+1)=620, total=893
# - 3840×640 image (6×1 tiles): global=273, local=(1×10)×(6×10+1)=610, total=883
```

**Token Layout Visualization (Gundam global view, 1024×1024)**:
```
Row 0:  [feat_0,0] [feat_0,1] ... [feat_0,15] [newline]    (17 tokens)
Row 1:  [feat_1,0] [feat_1,1] ... [feat_1,15] [newline]    (17 tokens)
...
Row 15: [feat_15,0] [feat_15,1] ... [feat_15,15] [newline] (17 tokens)
[view_separator]                                            (1 token)
                                                    Total: 17×16+1 = 273 tokens
```

**With Local Crops (Gundam mode)**:
```
[LOCAL CROPS: (h_tiles × 10) rows × (w_tiles × 10 + 1) cols]
[GLOBAL VIEW: 16 rows × 17 cols]
[view_separator]
```

#### Component 3: `collate_multimodal_batch`

Combines tokenization and vision outputs into training batch with **pre-training loss**:

```python
def collate_multimodal_batch(batch: List[dict]) -> dict:
    """
    Collate a batch of samples into padded tensors.
    Uses PRE-TRAINING LOSS: supervise all text tokens.

    Each sample has:
        - ids, image_positions (from render_vision_pretraining)
        - pixel_values, crops, spatial_info (from process_images, or None)

    Returns:
        inputs: (B, T) - padded token IDs
        targets: (B, T) - targets with -1 ONLY at image positions
        pixel_values: (B, N, 3, H, W) or None
        images_seq_mask: (B, T) - True at image token positions
    """
    # ... pad sequences ...

    for i, sample in enumerate(batch):
        ids = torch.tensor(sample["ids"], dtype=torch.long)
        n = len(ids)
        inputs[i, :n-1] = ids[:-1]

        # Pre-training loss: supervise ALL text tokens
        targets[i, :n-1] = ids[1:]

        # Only mask image placeholder positions
        for start, end in sample.get("image_positions", []):
            targets[i, start:end-1] = -1  # -1 for input/target shift

    return {"inputs": inputs, "targets": targets, "pixel_values": pixel_values, ...}
```

#### Data Generator Flow (Plain Prompt Only)

```python
sample = dataset[i]

# === ADAPTER: Normalize format at boundary ===
sample = normalize_sample(sample)  # {messages} -> {text, response}
validate_sample(sample)            # Fail fast on malformed data

# Build full text with <|assistant_end|> stop token
# This teaches the model when to stop generating (consistent with nanochat inference)
full_text = f"{sample['text']}{sample['response']}<|assistant_end|>"

if "image_path" in sample:
    # Vision data - all use plain prompt format
    image = Image.open(sample["image_path"])
    vision_output = processor.process_images([image], mode=resolution_mode)
    num_tokens = vision_output["num_tokens_per_image"]

    ids, img_pos = tokenizer.render_vision_pretraining(
        full_text, image_token_counts=num_tokens
    )

    batch.append({
        "ids": ids, "image_positions": img_pos,
        "pixel_values": vision_output["pixel_values"],
        "images_crop": vision_output["images_crop"],
        "images_spatial_crop": vision_output["images_spatial_crop"],
    })
else:
    # Text-only data (Stage 2)
    ids = [bos_token_id] + tokenizer.encode(full_text)
    batch.append({"ids": ids, "image_positions": [], "pixel_values": None})

# Collate when batch is full
yield collate_multimodal_batch(batch)
```

Key points:
1. **Single tokenization path**: `render_vision_pretraining()` for ALL vision data (no conversation markers)
2. **Pre-training loss**: Supervise ALL text tokens
3. **Only image positions masked**: `targets[image_positions] = -1`
4. **Multi-resolution support**: Pass `mode` to `process_images()` for Tiny/Small/Base/Large/Gundam
5. **`image_token_id`** must be added to the tokenizer vocabulary

### 5.2 Batch Structure Change

**Current:**
```python
inputs: (B, T)   # Token IDs
targets: (B, T)  # Target token IDs
```

**New:**
```python
inputs: (B, T)                    # Token IDs (with image placeholder tokens)
targets: (B, T)                   # Target token IDs
pixel_values: (B, N, 3, H, W)     # Preprocessed images
images_seq_mask: (B, T)           # Boolean: which positions are vision tokens
```

### 5.3 Model Forward Change

**Current (nanochat):**
```python
logits = model(input_ids)
loss = F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=-1)
```

**New (vision-enabled):**
```python
# 1. Encode images through DeepEncoder
vision_embeds = deep_encoder(pixel_values)  # (B, num_vision_tokens, n_embd)

# 2. Get text embeddings
text_embeds = gpt.transformer.wte(input_ids)  # (B, T, n_embd)

# 3. Merge vision embeddings at <image> token positions
inputs_embeds = merge_multimodal_embeddings(
    input_ids, text_embeds, vision_embeds, image_token_id
)

# 4. Forward through GPT (same as nanochat)
logits = gpt(inputs_embeds=inputs_embeds)

# 5. Loss (identical to nanochat)
loss = F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=-1)
```

### 5.4 Collate Function Extension

```python
def collate_vision_batch(batch):
    # Text part: same as nanochat
    max_seq_len = max(len(b['input_ids']) for b in batch)
    inputs = torch.full((B, max_seq_len), pad_token_id)
    targets = torch.full((B, max_seq_len), -1)

    for i, sample in enumerate(batch):
        n = len(sample['input_ids'])
        inputs[i, :n] = sample['input_ids']
        targets[i, :n-1] = sample['input_ids'][1:]
        # Mask user turns
        targets[i, sample['mask'] == 0] = -1

    # Vision part: new
    if any(b['pixel_values'] is not None for b in batch):
        pixel_values = torch.stack([b['pixel_values'] for b in batch])
        images_seq_mask = torch.stack([b['images_seq_mask'] for b in batch])
    else:
        pixel_values = None
        images_seq_mask = None

    return inputs, targets, pixel_values, images_seq_mask
```

### 5.5 Refactoring `tokenize_with_images` into `VisionProcessor`

**Key insight**: Extract the image processing logic from `tokenize_with_images()` into a standalone `VisionProcessor.process_images()` method. This separates image → tensor conversion from tokenization.

From `image_process.py` (lines 330-448), extract these responsibilities:

| Keep in `process_images` | Move to `render_vision_pretraining` |
|--------------------------|------------------------------|
| Dynamic tiling/cropping | `<image>` placeholder expansion |
| Image transforms (normalize, resize) | Plain prompt concatenation |
| Global + local view creation | BOS token insertion |
| Token count calculation | BOS/EOS token insertion |

```python
class VisionProcessor:
    """Extracted from DeepseekOCRProcessor - handles only image processing."""

    def process_images(self, images: List[Image.Image]) -> dict:
        """
        Process images into tensors. No tokenization.

        Returns:
            pixel_values: (N, 3, 1024, 1024) - global views
            images_crop: (N, num_crops, 3, 640, 640) - local crops
            images_spatial_crop: (N, 2) - tile layout per image
            num_tokens_per_image: List[int] - tokens needed per image
        """
        all_pixel_values = []
        all_crops = []
        all_spatial = []
        all_num_tokens = []

        for image in images:
            # Dynamic tiling (from tokenize_with_images lines 380-420)
            if image.width > 640 or image.height > 640:
                crops, spatial = self.dynamic_preprocess(image)
            else:
                crops, spatial = [image], (1, 1)

            # Transform to tensors
            global_view = self.transform(self.pad_to_square(image, 1024))
            crop_tensors = torch.stack([self.transform(c) for c in crops])

            # Calculate token count
            num_tokens = self.calculate_num_tokens(spatial)

            all_pixel_values.append(global_view)
            all_crops.append(crop_tensors)
            all_spatial.append(spatial)
            all_num_tokens.append(num_tokens)

        return {
            "pixel_values": torch.stack(all_pixel_values),
            "images_crop": all_crops,  # List, variable length per image
            "images_spatial_crop": all_spatial,
            "num_tokens_per_image": all_num_tokens,
        }
```

#### Edge Case: Default Shapes When No Crops Exist

From reference code `image_process.py:484-494`, the dataloader must handle two edge cases:

**Case 1: Images exist but no crops needed (image ≤ 640×640)**

When an image is small enough that dynamic tiling produces `spatial = (1, 1)`, `images_crop_list` will be empty because the condition `num_width_tiles > 1 or num_height_tiles > 1` (line 398) is false.

```python
# Default when images_crop_list is empty:
images_crop = torch.zeros((1, 3, image_size, image_size)).unsqueeze(0)  # Shape: (1, 1, 3, 640, 640)
```

**Case 2: No images at all (text-only sample)**

```python
# Defaults when images_list is empty:
pixel_values = torch.zeros((1, 3, base_size, base_size))      # Shape: (1, 3, 1024, 1024)
images_spatial_crop = torch.zeros((1, 1), dtype=torch.long)   # Shape: (1, 1) ⚠️ different from (N, 2)
images_crop = torch.zeros((1, 3, image_size, image_size)).unsqueeze(0)  # Shape: (1, 1, 3, 640, 640)
```

**Summary Table:**

| Tensor | Normal (with crops) | No Crops (small image) | No Images (text-only) |
|--------|---------------------|------------------------|----------------------|
| `pixel_values` | `(N, 3, 1024, 1024)` | `(N, 3, 1024, 1024)` | `(1, 3, 1024, 1024)` zeros |
| `images_spatial_crop` | `(N, 2)` | `(N, 2)` with `[1, 1]` values | `(1, 1)` zeros ⚠️ |
| `images_crop` | `(1, num_crops, 3, 640, 640)` | `(1, 1, 3, 640, 640)` zeros | `(1, 1, 3, 640, 640)` zeros |

**⚠️ Shape Inconsistency Note:** `images_spatial_crop` has shape `(1, 1)` when no images exist, but `(N, 2)` otherwise. The model's forward pass must handle this inconsistency (or we normalize it to always be `(N, 2)` in our implementation).

**Implementation in `VisionProcessor`:**

```python
def process_images(self, images: List[Image.Image]) -> dict:
    # ... processing logic ...

    # Handle edge cases for consistent tensor shapes
    if len(all_pixel_values) == 0:
        # No images - return zero tensors
        return {
            "pixel_values": torch.zeros((1, 3, self.base_size, self.base_size)),
            "images_crop": torch.zeros((1, 1, 3, self.image_size, self.image_size)),
            "images_spatial_crop": torch.zeros((1, 2), dtype=torch.long),  # Normalized to (1, 2)
            "num_tokens_per_image": [0],
        }

    pixel_values = torch.stack(all_pixel_values)
    images_spatial_crop = torch.tensor(all_spatial, dtype=torch.long)  # (N, 2)

    # Stack crops or return zero tensor if no crops
    if any(len(crops) > 0 for crops in all_crops):
        # Flatten all crops into single tensor
        all_crop_tensors = []
        for crops in all_crops:
            all_crop_tensors.extend(crops if isinstance(crops, list) else [crops])
        images_crop = torch.stack(all_crop_tensors).unsqueeze(0)  # (1, total_crops, 3, 640, 640)
    else:
        images_crop = torch.zeros((1, 1, 3, self.image_size, self.image_size))

    return {
        "pixel_values": pixel_values,
        "images_crop": images_crop,
        "images_spatial_crop": images_spatial_crop,
        "num_tokens_per_image": all_num_tokens,
    }
```

This approach:
1. **Uses `render_vision_pretraining`** for plain prompt tokenization
2. **Keeps `VisionProcessor` focused** on image → tensor conversion
3. **Enables clean testing** - each component testable in isolation
4. **Matches VLM patterns** used by LLaVA, Qwen-VL, InternVL

---

## 6. Multi-Resolution Training

**Following Paper Section 3.2.2**: Support multiple resolution modes via dynamic positional encoding interpolation.

**CRITICAL**: Training must support ALL modes, not just Gundam. The reference `config.py` hardcodes `BASE_SIZE=1024, IMAGE_SIZE=640, CROP_MODE=True` for inference, but training should cover all modes.

### Mode Configuration Table

| Mode | base_size | image_size | crop_mode | Tokens |
|------|-----------|------------|-----------|--------|
| Tiny | 512 | 512 | False | 73 |
| Small | 640 | 640 | False | 111 |
| Base | 1024 | 1024 | False | 273 |
| Large | 1280 | 1280 | False | 421 |
| Gundam | 1024 | 640 | True | 273 + local |

**Token counts include newline (1 per row) + view_separator (1 per view).** README's 64/100/256/400 are feature-only counts. **Use these full counts for batching/max_seq_len calculations.**

### Gundam Local Crop Token Budget (MIN_CROPS=2, MAX_CROPS=6)

With `num_queries_local = 640/16/4 = 10`, local tokens = `(h_tiles × 10) × (w_tiles × 10 + 1)`:

| Tiles (w×h) | Total Tiles | Local Tokens | Total (273 + local) |
|-------------|-------------|--------------|---------------------|
| 2×1 | 2 | 210 | 483 |
| 1×2 | 2 | 220 | 493 |
| 3×1 | 3 | 310 | 583 |
| 1×3 | 3 | 330 | 603 |
| 2×2 | 4 | 420 | 693 |
| 4×1 | 4 | 410 | 683 |
| 1×4 | 4 | 440 | 713 |
| 5×1 | 5 | 510 | 783 |
| 1×5 | 5 | 550 | 823 |
| 3×2 | 6 | 620 | 893 |
| 2×3 | 6 | 630 | 903 |
| 6×1 | 6 | 610 | 883 |
| 1×6 | 6 | 660 | 933 |

**Max vision tokens with MAX_CROPS=6: 933** (1×6 layout). Plan `max_seq_len` accordingly: 933 vision + text prompt + response.

### Mode Sampling for Training

To exercise all resolution paths during training, use a **mode sampler** rather than purely adaptive selection:

```python
import random

MODES = {
    "tiny":   {"base_size": 512,  "image_size": 512,  "crop_mode": False, "tokens": 73},
    "small":  {"base_size": 640,  "image_size": 640,  "crop_mode": False, "tokens": 111},
    "base":   {"base_size": 1024, "image_size": 1024, "crop_mode": False, "tokens": 273},
    "large":  {"base_size": 1280, "image_size": 1280, "crop_mode": False, "tokens": 421},
    "gundam": {"base_size": 1024, "image_size": 640,  "crop_mode": True,  "tokens": "273+local"},
}

def sample_mode_for_image(width: int, height: int, rng: random.Random) -> dict:
    """
    Sample a mode for training. Ensures all modes are exercised.

    Strategy:
    - Small images (≤640): sample from Tiny/Small/Base/Large (upsample to larger modes)
    - Large images (>640): sample from Base/Large/Gundam (pad or tile)
    """
    if width <= 640 and height <= 640:
        # Small images: can use any fixed mode (will be upsampled/padded)
        mode_name = rng.choice(["tiny", "small", "base", "large"])
    else:
        # Large images: use Base/Large (pad, no crops) or Gundam (tile)
        mode_name = rng.choice(["base", "large", "gundam"])

    return MODES[mode_name]
```

**Processing per mode:**
- **Fixed modes (Tiny/Small/Base/Large)**: Resize/pad image to `base_size × base_size`, no local crops
- **Gundam mode**: Pad to 1024×1024 global + `dynamic_preprocess()` for 640×640 local tiles

For Gundam mode, use `dynamic_preprocess()` from reference code to get the tile layout based on aspect ratio. The tile search uses `MIN_CROPS=2, MAX_CROPS=6` by default.

### Token Count Formula

**Note**: Use the detailed `count_image_tokens()` formula in Section 5.1 for accurate counts. The formula is:
- Global tokens: `(num_queries + 1) * num_queries + 1` where `num_queries = resolution / patch_size / downsample_ratio`
- For 1024×1024 global view: `(16+1)*16+1 = 273` tokens
- Local crops add: `(h_tiles * num_queries) * (w_tiles * num_queries + 1)` tokens

### Dynamic Positional Encoding (from DeepSeek-OCR reference code)

**SAM** uses `get_abs_pos()` from `deepencoder/sam_vary_sdpa.py:19-38`:

```python
def get_abs_pos(abs_pos, tgt_size):
    """Interpolate absolute positional embeddings to target size.

    Args:
        abs_pos: (1, H, W, dim) - original positional embeddings
        tgt_size: int - target spatial size (H=W after patch embedding)

    Returns:
        (1, tgt_size, tgt_size, dim) - interpolated positional embeddings
    """
    dtype = abs_pos.dtype
    src_size = abs_pos.size(1)
    if src_size != tgt_size:
        old_pos_embed = abs_pos.permute(0, 3, 1, 2).to(torch.float32)
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode='bicubic',
            antialias=True,
            align_corners=False,
        ).to(dtype)
        return new_pos_embed.permute(0, 2, 3, 1)
    return abs_pos
```

**CLIP** uses `get_abs_pos()` from `deepencoder/clip_sdpa.py:63-99` (handles CLS token separately):

```python
def get_abs_pos(abs_pos, tgt_size):
    """Interpolate CLIP positional embeddings, preserving CLS token.

    Args:
        abs_pos: (1, L+1, dim) - original positional embeddings (CLS + patches)
        tgt_size: int - target number of patches (will be sqrt'd)

    Returns:
        (1, tgt_size+1, dim) - interpolated positional embeddings
    """
    dim = abs_pos.size(-1)
    abs_pos_new = abs_pos.squeeze(0)
    cls_token, old_pos_embed = abs_pos_new[:1], abs_pos_new[1:]  # Split CLS token

    src_size = int(math.sqrt(abs_pos_new.shape[0] - 1))
    tgt_size = int(math.sqrt(tgt_size))

    if src_size != tgt_size:
        old_pos_embed = old_pos_embed.view(1, src_size, src_size, dim).permute(0, 3, 1, 2)
        old_pos_embed = old_pos_embed.to(torch.float32)
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode='bicubic',
            antialias=True,
            align_corners=False,
        ).to(abs_pos.dtype)
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1).view(tgt_size * tgt_size, dim)
        vision_pos_embed = torch.cat([cls_token, new_pos_embed], dim=0)
        return vision_pos_embed.view(1, tgt_size * tgt_size + 1, dim)
    return abs_pos
```

### Training Strategy

Use `find_closest_aspect_ratio()` from [image_process.py](../reference_code/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/image_process.py) to determine mode:

```python
def select_resolution_mode(image: Image.Image) -> Tuple[str, Tuple[int, int]]:
    """
    Determine resolution mode based on image dimensions.
    Uses find_closest_aspect_ratio() for Gundam mode tile calculation.

    Returns:
        mode: "small" | "base" | "gundam"
        tiles: (width_tiles, height_tiles) for Gundam, (1, 1) otherwise
    """
    width, height = image.size

    # Small images: resize to 640×640, no tiling
    if width <= 640 and height <= 640:
        return "small", (1, 1)

    # Larger images: use find_closest_aspect_ratio to get tile layout
    # This is what count_tiles() does internally
    aspect_ratio = width / height
    target_ratios = set(
        (i, j) for n in range(MIN_CROPS, MAX_CROPS + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if MIN_CROPS <= i * j <= MAX_CROPS
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    tiles = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, width, height, image_size=640
    )

    if tiles == (1, 1):
        return "base", (1, 1)  # 1024×1024 global view only
    else:
        return "gundam", tiles  # Dynamic tiling + global view

# Then use dynamic_preprocess() for Gundam mode to get actual crops:
if mode == "gundam":
    crops, tiles = dynamic_preprocess(image, image_size=640)
    # crops: List[PIL.Image] - 640×640 tiles
    # tiles: (width_tiles, height_tiles)
```

**Mode Selection Logic:**

| Image Size | `find_closest_aspect_ratio` Result | Mode | Processing |
|------------|-----------------------------------|------|------------|
| ≤640×640 | N/A (skip) | Small | resize to 640×640 |
| 641-1024 square | (1, 1) | Base | pad to 1024×1024 |
| Wide/tall >640 | (2, 1), (1, 2), (2, 2), etc. | Gundam | `dynamic_preprocess()` + 1024 global |

**Batch Construction:**
1. Sample images from dataset
2. Call `select_resolution_mode()` to determine mode and tiles
3. For Gundam mode, use `dynamic_preprocess()` to get 640×640 crops
4. Calculate token count using the tile layout
5. Batch samples with same token count together (or pad to max)

---

## 7. Plain Prompt Format

**See [data_plan.md](data_plan.md)** for complete details on:
- Target data formats (`{text, response, image_path}`)
- Prompt examples from the paper
- Loss masking strategy (supervise all text tokens, mask only image positions)
- Dataset transformations

---

## 8. Optimizer Setup Summary

### Stage 1 (All Components Trainable)

```python
# Matrix parameters (linear layers) - Muon
matrix_params = [
    *sam_encoder.parameters(),
    *conv_compressor.parameters(),
    *clip_encoder.parameters(),
    *projector.parameters(),
    *gpt.transformer.h.parameters(),
]
muon_optim = Muon(matrix_params, lr=0.02, momentum=0.95)

# Embedding/scalar parameters - AdamW
embedding_params = [
    image_newline, view_separator,
    *gpt.transformer.wte.parameters(),
    gpt.lm_head.weight,
]
adam_optim = AdamW(embedding_params, lr=5e-5, betas=(0.9, 0.95), weight_decay=0.1)

# Cosine annealing scheduler
scheduler = CosineAnnealingLR(total_steps, warmup_steps=2000)
```

### Stage 2 (SAM+Compressor Frozen)

```python
# Freeze SAM and Conv compressor
for p in sam_encoder.parameters():
    p.requires_grad = False
for p in conv_compressor.parameters():
    p.requires_grad = False

# Matrix parameters - Muon
matrix_params = [
    *clip_encoder.parameters(),
    *projector.parameters(),
    *gpt.transformer.h.parameters(),
]
muon_optim = Muon(matrix_params, lr=0.02, momentum=0.95)

# Embedding parameters - AdamW
embedding_params = [
    image_newline, view_separator,
    *gpt.transformer.wte.parameters(),
    gpt.lm_head.weight,
]
adam_optim = AdamW(embedding_params, lr=3e-5, betas=(0.9, 0.95), weight_decay=0.1)

# Step-based scheduler
scheduler = MultiStepLR(milestones=[...], gamma=0.1)
```

---

## 9. Dataloader Redesign: Why Streaming Packing Won't Work

### The Problem with nanochat's `mid_data_generator`

nanochat's dataloader ([mid_train.py:117-156](../reference_code/nanochat/scripts/mid_train.py)) uses **streaming token packing**:

```python
token_buffer = deque()
while len(token_buffer) < needed_tokens:
    conversation = dataset[cursor]
    ids, _ = tokenizer.render_conversation(conversation)
    token_buffer.extend(ids)  # <- tokens from multiple samples concatenated
# Sample boundaries are LOST when chunking into batches
```

This destroys sample boundaries. A batch might contain `[...end_of_A][all_of_B][start_of_C...]`, making it impossible to align `pixel_values` with their corresponding `<image>` token positions.

### Solution: Discrete Sample Batching

Replace the streaming packer with **discrete per-sample batching** (standard VLM approach used by LLaVA, Qwen-VL):

```python
from nano_dpsk_ocr.data.sample_adapter import normalize_sample, validate_sample

def vision_data_generator(split):
    """Drop-in replacement for mid_data_generator. Yields 6-tuple per batch."""
    global last_step, approx_progress
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    cursor, it = ddp_rank, 0

    while True:
        batch_samples = []
        for _ in range(device_batch_size):
            sample = dataset[cursor]
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor -= dataset_size
                if split == "train": last_step = True

            # === ADAPTER: Normalize format at boundary ===
            sample = normalize_sample(sample)  # {messages} -> {text, response}
            validate_sample(sample)            # Fail fast on malformed data

            if "image_path" in sample:
                image = Image.open(sample["image_path"])
                vision_out = processor.process_images([image])
                ids, img_pos = tokenizer.render_vision_pretraining(
                    sample["text"], sample["response"],
                    image_token_counts=vision_out["num_tokens_per_image"])
                crops = vision_out.get("images_crop")
                spatial = vision_out.get("images_spatial_crop")
                batch_samples.append({
                    "ids": ids, "image_positions": img_pos,
                    "pixel_values": vision_out["pixel_values"][0],
                    "images_crop": crops[0] if crops is not None else None,
                    "images_spatial_crop": spatial[0] if spatial is not None else None,
                })
            else:
                ids = [bos_id] + tokenizer.encode(sample["text"] + sample["response"])
                batch_samples.append({"ids": ids, "image_positions": [],
                    "pixel_values": None, "images_crop": None, "images_spatial_crop": None})

        it += 1
        if split == "train":
            approx_progress = it / num_iterations if num_iterations > 0 else cursor / dataset_size
            if num_iterations > 0 and it >= num_iterations: last_step = True

        batch = collate_vision_batch(batch_samples, max_seq_len, pad_token_id, bos_id)
        yield tuple(v.to(device) if v is not None else None for v in batch.values())
```

### `collate_vision_batch` Implementation

```python
def collate_vision_batch(samples, max_seq_len, pad_token_id, bos_id):
    """Collate discrete samples into padded batch tensors."""
    B = len(samples)
    inputs = torch.full((B, max_seq_len), pad_token_id, dtype=torch.int32)
    targets = torch.full((B, max_seq_len), -1, dtype=torch.int64)
    images_seq_mask = torch.zeros((B, max_seq_len), dtype=torch.bool)
    pixel_list, crop_list, spatial_list = [], [], []

    for i, sample in enumerate(samples):
        ids = sample["ids"]
        assert ids[0] == bos_id, f"Sample {i}: ids must start with BOS"
        n = min(len(ids), max_seq_len + 1)
        ids = ids[:n]

        inputs[i, :n-1] = torch.tensor(ids[:-1], dtype=torch.int32)
        targets[i, :n-1] = torch.tensor(ids[1:], dtype=torch.int64)

        # Mask image positions (targets[k] predicts ids[k+1], so mask targets[start-1:end-1])
        for start, end in sample.get("image_positions", []):
            start, end = max(0, start), min(end, n)
            if start < end:
                targets[i, max(0,start-1):min(end-1,max_seq_len)] = -1
                images_seq_mask[i, start:min(end,max_seq_len)] = True

        pixel_list.append(sample["pixel_values"])
        crop_list.append(sample["images_crop"])
        spatial_list.append(sample["images_spatial_crop"])

    # Stack vision tensors with zero-padding for None entries
    pixel_values = _stack_with_padding(pixel_list)
    images_crop = _stack_with_padding(crop_list, pad_variable_dim=0)  # variable num_crops
    images_spatial_crop = _stack_with_padding(spatial_list, default_val=1)  # default (1,1)

    return {"inputs": inputs, "targets": targets, "pixel_values": pixel_values,
            "images_crop": images_crop, "images_spatial_crop": images_spatial_crop,
            "images_seq_mask": images_seq_mask}


def _stack_with_padding(tensors, pad_variable_dim=None, default_val=0):
    """Stack tensors, zero-padding None entries. Handles variable-size dim if specified."""
    non_none = [t for t in tensors if t is not None]
    if not non_none:
        return None
    ref = non_none[0]

    # Pad variable dimension (e.g., num_crops) to max across batch
    if pad_variable_dim is not None:
        max_size = max(t.shape[pad_variable_dim] for t in non_none)
        padded = []
        for t in non_none:
            if t.shape[pad_variable_dim] < max_size:
                pad_shape = list(t.shape)
                pad_shape[pad_variable_dim] = max_size - t.shape[pad_variable_dim]
                t = torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)], dim=pad_variable_dim)
            padded.append(t)
        non_none = padded

    # Fill None entries with zeros (or default_val for spatial)
    result = []
    j = 0
    for t in tensors:
        if t is not None:
            result.append(non_none[j])
            j += 1
        else:
            shape = list(non_none[0].shape)
            result.append(torch.full(shape, default_val, dtype=non_none[0].dtype))
    return torch.stack(result)
```

### Training Loop Changes

```python
# OLD:  x, y = next(train_loader)
# NEW:
x, y, pixel_values, images_crop, images_spatial_crop, images_seq_mask = next(train_loader)
loss = model(x, y, pixel_values=pixel_values, images_crop=images_crop,
             images_spatial_crop=images_spatial_crop, images_seq_mask=images_seq_mask)
```

### Edge Cases

- **Text-only batch**: All vision tensors `None`, `images_seq_mask` all False
- **Mixed batch**: `_stack_with_padding` zero-pads None entries
- **Variable crops**: Padded to `max_crops` in batch via `pad_variable_dim=0`
- **Truncation**: `image_positions` clamped to truncated length

### `process_images` Contract

```python
# Returns dict with:
#   pixel_values: (N, 3, 1024, 1024) - global views
#   images_crop: (N, num_crops, 3, 640, 640) or None - local crops
#   images_spatial_crop: (N, 2) or None - [w_tiles, h_tiles]
#   num_tokens_per_image: List[int] - token counts for <image> expansion
```

### Why This Is Acceptable

The ~10-20% efficiency loss from padding (vs. streaming packing) is negligible because:
1. **Vision encoder dominates compute** - SAM+CLIP forward pass >> text embedding lookup
2. **Standard practice** - LLaVA, Qwen-VL, InternVL all use discrete batching
3. **Simpler debugging** - each sample's boundaries are explicit

---

## 10. Summary: What to Reuse vs. What to Change

| Component | Reuse from nanochat | Change Required |
|-----------|---------------------|-----------------|
| **gpt.py** | ⚠️ 95% | **CRITICAL**: Add `inputs_embeds` parameter (Section 5.0) |
| `TaskMixture` | ✅ 100% | None - works as-is (format-agnostic) |
| Task classes | ⚠️ 70% | Vision tasks: new `{text, response, image_path}` plain prompt format |
| `render_vision_pretraining` | 🆕 New | Plain prompt tokenization (no conversation markers) |
| `VisionProcessor` | 🆕 New | Extract from `tokenize_with_images()`, multi-resolution support |
| `collate_vision_batch` | 🆕 New | Discrete sample batching (replaces streaming packer) |
| **Data generator** | ❌ Replace | `vision_data_generator` replaces `mid_data_generator` (Section 9) |
| Training loop | ✅ 95% | Unpack 4 tensors instead of 2, pass to model |
| Loss function | ✅ 100% | None (mask image tokens only) |
| Optimizer structure | ✅ 90% | Add vision encoder param groups, Muon + AdamW split |
| Tokenizer vocab | ⚠️ 90% | Add `<image>`, `<|grounding|>` special tokens |
| DeepEncoder | ✅ 100% | Use as-is (includes `get_abs_pos()` for multi-res) |

---

## 11. Implementation Order

1. **Modify nanochat `gpt.py`** - Add `inputs_embeds` parameter (Section 5.0) **← PREREQUISITE**
2. **Add special tokens to tokenizer** - `<image>`, `<|grounding|>` etc.
3. **Create `render_vision_pretraining()`** - plain prompt tokenization with `<image>` expansion
4. **Create `VisionProcessor`** - extract image processing from `tokenize_with_images()`, multi-resolution support
5. **Create `collate_multimodal_batch`** - combine tokenization + vision outputs into batch
6. **Create vision task classes** - use `{text, response, image_path}` format (plain prompts)
7. **Modify training loop** - add vision embedding merge step, support variable seq lengths (4096/8192)
8. **Add Stage 1 training** - all components trainable, seq_len=4096, cosine scheduler
9. **Add Stage 2 training** - freeze SAM + Conv, seq_len=8192, step scheduler
10. **Add multi-resolution mode selection** - Tiny/Small/Base/Large/Gundam (uses existing `get_abs_pos()` in DeepEncoder)
11. **Test on small data first** - overfit to 20 samples

---

## 12. Key Architectural Notes

### Why SAM + Conv are frozen in Stage 2

1. **SAM produces local features** - after Stage 1, SAM already extracts good patch-level features
2. **Conv is a simple reducer** - just spatial downsampling, no semantic learning needed
3. **CLIP needs global reasoning** - benefits from seeing more diverse data with LLM feedback
4. **Compute efficiency** - fewer trainable params = faster training, larger batch sizes

**See [overview.md](overview.md#model-architecture)** for the full DeepEncoder architecture diagram.
