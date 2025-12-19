# Vision & Multimodal Training

Karpathy-style Task pattern for training VLMs with DeepSeek-OCR algorithm.

## Data Format (Unified Conversation Format)

All tasks (vision and text) use the same conversation format for multimodal scalability:

```python
# Vision sample
{
    "messages": [
        {"role": "user", "content": "<image>\nOCR this document."},
        {"role": "assistant", "content": "Hello world"}
    ],
    "image_path": "data/images/doc.png",  # optional
}

# Text sample (same format, no media paths)
{
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"}
    ]
}
```

| Field | Description |
|-------|-------------|
| `messages` | Conversation in chat format (user/assistant roles) |
| `image_path` | Optional path to image file |
| `audio_path` | Optional path to audio file (future) |
| `video_path` | Optional path to video file (future) |

This unified format:
- Scales to audio/video without changing the loader
- Allows mixing vision + text in single TaskMixture
- ALL samples use `tokenizer.render_conversation()` - one tokenization path for all modalities

## Datasets

### Vision Datasets (~3M total)

| Dataset | HuggingFace ID | Task | Size |
|---------|----------------|------|------|
| DocBank | [`liminghao1630/DocBank`](https://huggingface.co/datasets/liminghao1630/DocBank) | Document OCR | 400K |
| olmOCR | [`allenai/olmOCR-mix-1025`](https://huggingface.co/datasets/allenai/olmOCR-mix-1025) | PDF → Markdown | 270K |
| LLaVA-CC3M | [`liuhaotian/LLaVA-CC3M-Pretrain-595K`](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K) | Image description | 595K |
| PlotQA | [`achang/plot_qa`](https://huggingface.co/datasets/achang/plot_qa) | Chart description | 157K |
| ChartQA | [`HuggingFaceM4/ChartQA`](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) | Chart Q&A | 28K |
| FigureQA | [`vikhyatk/figureqa`](https://huggingface.co/datasets/vikhyatk/figureqa) | Yes/No reasoning | 100K |
| PubTables-1M | [`bsmock/pubtables-1m`](https://huggingface.co/datasets/bsmock/pubtables-1m) | Table → HTML | 948K |
| LaTeX-Formulas | [`OleehyO/latex-formulas`](https://huggingface.co/datasets/OleehyO/latex-formulas) | Printed math → LaTeX | 550K |

### Text Datasets (~570K total, from mid_train.py)

| Dataset | HuggingFace ID | Task | Size |
|---------|----------------|------|------|
| SmolTalk | [`HuggingFaceTB/smoltalk`](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | General conversations | 460K |
| MMLU | [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu) | Multiple choice (ARC, MC_TEST, OBQA, RACE) | 100K |
| GSM8K | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) | Math + calculator tool use | 8K |
| CustomJSON | local | Identity conversations | 2K |

Combined total: ~3.6M samples (84% vision, 16% text)

## Loss Masking Strategy

**Mid-training style** (following DeepSeek-OCR):
```
Sequence: [IMG×273, \n, O, C, R, ..., response_tokens, PAD...]
Targets:  [-1×273,  \n, O, C, R, ..., response_tokens, -1...]  ← train on ALL text
```

- Train on ALL text tokens (not just response)
- Mask media token positions AND padding positions
- ALL samples use `render_conversation()` - handles media placeholders automatically

```python
# Initialize all targets as masked (-1)
targets_batch = torch.full((B, T), -1, dtype=torch.long)

# Fill only valid positions
targets_batch[b, :seq_len] = torch.tensor(ids[1:seq_len + 1])

# Mask image token positions
targets_batch[b, inputs_batch[b] == image_token_id] = -1
```

This matches DeepSeek-OCR paper: "not a chatbot due to absent SFT stage"

## Task Pattern

All tasks return unified conversation format:

```python
class OlmOCR(Task):
    def __init__(self, data_dir, split="train"):
        self.ds = load_dataset("allenai/olmOCR-mix-1025", split=split)
        self.data_dir = data_dir

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        return {
            "messages": [
                {"role": "user", "content": "<image>\nOCR this document."},
                {"role": "assistant", "content": row["text"]}
            ],
            "image_path": os.path.join(self.data_dir, row["image_path"]),
        }

# Text tasks use same format (no image_path)
class GSM8K(Task):
    def get_example(self, index):
        row = self.ds[index]
        return {
            "messages": [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]}
            ]
        }
```

## Unified TaskMixture

Mix vision and text tasks in single TaskMixture:

```python
from tasks.common import TaskMixture

# Everything in one mixture
train_ds = TaskMixture([
    # Vision tasks (all 8 datasets, ~3M total)
    DocBank(data_dir="data/docbank"),           # 400K document OCR
    OlmOCR(data_dir="data/olmocr"),             # 270K PDF → Markdown
    LLaVACC3M(data_dir="data/llava_cc3m"),      # 595K image description
    PlotQA(data_dir="data/plotqa"),             # 157K chart description
    ChartQA(data_dir="data/chartqa"),           # 28K chart Q&A
    FigureQA(data_dir="data/figureqa"),         # 100K yes/no reasoning
    PubTables(data_dir="data/pubtables"),       # 948K table → HTML
    LaTeXFormulas(data_dir="data/latex"),       # 550K printed math → LaTeX
    # Text tasks (from mid_train.py, ~570K total)
    SmolTalk(split="train"),                    # 460K general conversations
    MMLU(subset="auxiliary_train", split="train"),  # 100K multiple choice (ARC, MC_TEST, OBQA, RACE)
    GSM8K(subset="main", split="train"),        # 8K math + calculator tool use
    CustomJSON(filepath=identity_filepath),     # 1K identity conversations
    CustomJSON(filepath=identity_filepath),     # 1K identity (2x oversample)
])

# One loader handles all modalities
for inputs, targets, media in multimodal_data_generator(train_ds, tokenizer, ...):
    # media["pixel_values"] is None for text-only batches
    loss = model(input_ids=inputs, targets=targets, pixel_values=media["pixel_values"])
```

## Why Unified Approach?

Separate loaders per modality don't scale:

| Modalities | Separate Loaders | Unified TaskMixture |
|------------|------------------|---------------------|
| 2 (text + vision) | 2 loaders, coin flip | 1 loader |
| 3 (+ audio) | 3 loaders, weighted sample | 1 loader |
| 4 (+ video) | 4 loaders, complex mixing | 1 loader |
| Multi-modal (image + audio) | **Can't handle** | Natural |

Key insight: **modality is a property of the sample, not the loader.**

Text data should be SFT-style (SmolTalk, GSM8K, MMLU) rather than raw web text (FineWeb).
This aligns with VILA paper finding: "re-blending text-only instruction data boosts VLM task accuracy."

## Current Implementation (Legacy)

### Current Data Format (vision_dataloader.py)
The current vision loader uses a simple prompt/answer JSON format:

```python
# data/train.json format:
{"prompt": "<image>\nOCR this.", "answer": "Hello", "image": "images/doc.png"}
```

### Current Loaders (vis_mid_train.py)
```python
# Vision: PyTorch DataLoader via create_vision_loader() (vision_dataloader.py:76)
vision_loader = create_vision_loader(batch_size, seq_len, data_dir, tokenizer, "train", base_size)

# Text: Streaming parquet via tokenizing_distributed_data_loader() (dataloader.py)
text_loader = tokenizing_distributed_data_loader(B=batch_size, T=seq_len, split="train", device=device)

# Mixing at step level with random coin flip (vis_mid_train.py:261-272)
if text_loader is not None and random.random() < text_ratio:
    inputs, targets = next(text_loader)
    pixel_values = None
else:
    inputs, targets, pixel_values = next(vision_iter)
```

### Text SFT Loaders (mid_train.py, chat_sft.py)
Text tasks use conversation format with TaskMixture:
```python
train_dataset = TaskMixture([SmolTalk(...), MMLU(...), GSM8K(...)])
# mid_data_generator yields (inputs, targets) by calling:
#   tokenizer.render_conversation(conversation)
```

---

## Migration Plan: Unified Multimodal Pipeline

### Goal
One tokenizer, one loader for all stages. Change what's in TaskMixture, not the code.

- **Stage 1**: `TaskMixture([VisionTask()])`
- **Stage 2**: `TaskMixture([VisionTask(), TextTask(), ...])`

### Step 1: Update `nanochat/tokenizer.py` for Media Tokens

Add media placeholder support to `render_conversation()`. This is the key insight:
`encode_ordinary()` skips special tokens (security feature), but we need `<|image|>` to emit.

```python
# Add to tokenizer.py at module level
MEDIA_PLACEHOLDERS = {
    "<image>": "<|image|>",
    "<audio>": "<|audio|>",   # future
    "<video>": "<|video|>",   # future
}

# In RustBPETokenizer.render_conversation(), update user message handling:
if message["role"] == "user":
    assert isinstance(content, str)
    # Handle media placeholders - replace and use allowed_special
    allowed = set()
    for placeholder, special in MEDIA_PLACEHOLDERS.items():
        if placeholder in content:
            content = content.replace(placeholder, special)
            allowed.add(special)

    if allowed:
        value_ids = list(self.enc.encode(content, allowed_special=allowed))
    else:
        value_ids = self.encode(content)

    add_tokens(user_start, 0)
    add_tokens(value_ids, 0)
    add_tokens(user_end, 0)
```

**Why this design:**
- One code path for ALL samples (text, vision, audio, video)
- Adding new modality = add one line to `MEDIA_PLACEHOLDERS`
- No separate `tokenize_with_image()` function needed
- Dataloader becomes modality-agnostic

### Step 2: Create `tasks/overfit_samples.py`

Vision task that returns conversation format:

```python
from tasks.common import Task
import json
import os

class OverfitSamples(Task):
    """Vision task - loads from data/{split}.json"""

    def __init__(self, data_dir="data", split="train", **kwargs):
        super().__init__(**kwargs)
        json_path = os.path.join(data_dir, f"{split}.json")
        with open(json_path) as f:
            self.samples = json.load(f)
        self.data_dir = data_dir

    def num_examples(self):
        return len(self.samples)

    def get_example(self, index):
        s = self.samples[index]
        return {
            "messages": [
                {"role": "user", "content": s["prompt"]},
                {"role": "assistant", "content": s["answer"]}
            ],
            "image_path": os.path.join(self.data_dir, s["image"]),
        }
```

### Step 3: Create `nanochat/multimodal_dataloader.py`

Per-sample processing with proper masking. Returns media dict for future scalability.
**Key simplification**: No `tokenize_with_image()` - all samples use `render_conversation()`.

```python
"""
Unified multimodal dataloader.

Usage:
    # Stage 1: vision only
    train_ds = TaskMixture([OverfitSamples(data_dir="data")])

    # Stage 2: vision + text
    train_ds = TaskMixture([OverfitSamples(...), SmolTalk(...), GSM8K(...)])

    # Same loader for both
    for inputs, targets, media in multimodal_data_generator(train_ds, ...):
        loss = model(input_ids=inputs, targets=targets, pixel_values=media["pixel_values"])
"""

import torch
import torch.distributed as dist
from PIL import Image
from nanochat.image_process import process_image, count_vision_tokens, expand_image_tokens


def multimodal_data_generator(dataset, tokenizer, B, T, base_size, device):
    """
    Yields (inputs, targets, media) batches.

    - media["pixel_values"] is None for text-only batches (model skips vision encoder)
    - targets initialized to -1, only valid positions filled (padding masked)
    - inputs padded with 0 (BOS token) - matches vision_dataloader.py behavior
    - DDP-aware via cursor striding
    """
    ddp = dist.is_initialized()
    rank = dist.get_rank() if ddp else 0
    world = dist.get_world_size() if ddp else 1

    n_img_tokens = count_vision_tokens(base_size)
    image_token_id = tokenizer.encode_special("<|image|>")

    cursor = rank
    while True:
        inputs_batch = torch.zeros(B, T, dtype=torch.long)
        targets_batch = torch.full((B, T), -1, dtype=torch.long)
        pixels_list = []
        batch_has_image = False

        for b in range(B):
            sample = dataset[cursor % len(dataset)]
            cursor += world

            # --- Tokenize (ONE path for all modalities) ---
            ids, _ = tokenizer.render_conversation(sample)

            # --- Handle image if present ---
            if "image_path" in sample and sample["image_path"]:
                ids = expand_image_tokens(ids, image_token_id, n_img_tokens)
                pixels = process_image(Image.open(sample["image_path"]), base_size)
                pixels_list.append(pixels)
                batch_has_image = True

            # --- Fill batch ---
            seq_len = min(len(ids) - 1, T)
            inputs_batch[b, :seq_len] = torch.tensor(ids[:seq_len], dtype=torch.long)
            targets_batch[b, :seq_len] = torch.tensor(ids[1:seq_len + 1], dtype=torch.long)
            targets_batch[b, inputs_batch[b] == image_token_id] = -1  # mask media tokens

        # --- Build media dict ---
        if batch_has_image:
            while len(pixels_list) < B:
                pixels_list.append(torch.zeros(3, base_size, base_size))
            media = {"pixel_values": torch.stack(pixels_list).to(device)}
        else:
            media = {"pixel_values": None}

        yield inputs_batch.to(device), targets_batch.to(device), media
```

**Design notes:**

1. **One tokenization path**: `render_conversation()` handles media placeholders via `MEDIA_PLACEHOLDERS` dict. No branching on modality.

2. **Scalable to new modalities**: To add audio/video, just add entries to `MEDIA_PLACEHOLDERS` in tokenizer.py and add processing logic here.

3. **Media dict pattern**: Returns `{"pixel_values": ...}`. Extend to `{"pixel_values": ..., "audio_features": ...}` when adding audio.

### Step 4: Update Training Scripts

Both stages use the same loader, just different TaskMixtures:

```python
# vis_tok_train.py (Stage 1) - MODIFY to use multimodal_dataloader
from nanochat.multimodal_dataloader import multimodal_data_generator
from tasks.overfit_samples import OverfitSamples
from tasks.common import TaskMixture

train_ds = TaskMixture([OverfitSamples(data_dir=data_dir)])
train_loader = multimodal_data_generator(train_ds, tokenizer, B, T, base_size, device)

for inputs, targets, media in train_loader:
    loss = model(input_ids=inputs, targets=targets, pixel_values=media["pixel_values"])
```

```python
# vis_mid_train.py (Stage 2) - MODIFY to use multimodal_dataloader
from nanochat.multimodal_dataloader import multimodal_data_generator
from tasks.overfit_samples import OverfitSamples
from tasks.smoltalk import SmolTalk
from tasks.common import TaskMixture

train_ds = TaskMixture([
    OverfitSamples(data_dir=data_dir),  # vision
    SmolTalk(split="train"),            # text
])
train_loader = multimodal_data_generator(train_ds, tokenizer, B, T, base_size, device)

for inputs, targets, media in train_loader:
    # media["pixel_values"] is None for text batches - model skips vision encoder
    loss = model(input_ids=inputs, targets=targets, pixel_values=media["pixel_values"])
```

### Files to Create/Modify

```
nanochat/
├── tokenizer.py              # MODIFY: add MEDIA_PLACEHOLDERS + encode_with_media_placeholders()
├── multimodal_dataloader.py  # NEW: ~50 lines (simplified - no tokenize_with_image)
├── vision_dataloader.py      # DEPRECATE after migration
├── dataloader.py             # DEPRECATE after migration
tasks/
├── overfit_samples.py        # NEW: ~25 lines
scripts/
├── vis_tok_train.py          # MODIFY: use multimodal_dataloader
├── vis_mid_train.py          # MODIFY: use multimodal_dataloader
```

### Key Design Decisions

1. **One tokenizer for all modalities**: `render_conversation()` handles media placeholders via `MEDIA_PLACEHOLDERS` dict. Adding new modality = one line.

2. **One loader for all stages**: Change TaskMixture contents, not the loader code.

3. **Media dict pattern**: Returns `{"pixel_values": tensor or None}`. None = model skips vision encoder. Scalable to `{"pixel_values": ..., "audio_features": ..., "video_frames": ...}`.

4. **Per-sample processing**: Each sample tokenized and padded individually. Preserves sample boundaries for vision.

5. **Proper masking**: `targets=-1` for padding AND media tokens. `inputs=0` (BOS) for padding.

6. **TaskMixture handles mixing**: Ratio control via task repetition (e.g., add SmolTalk twice for 2x weight).

### Masking Strategy

Train on ALL text tokens, mask only media + padding positions:

```python
targets_batch = torch.full((B, T), -1, ...)  # all masked initially
targets_batch[b, :seq_len] = ids[1:seq_len+1]  # fill valid positions
# Mask all media token positions (image, audio, video, etc.)
for media_token_id in [image_token_id, audio_token_id, ...]:
    targets_batch[b, inputs_batch[b] == media_token_id] = -1
```

This matches DeepSeek-OCR paper: "not a chatbot due to absent SFT stage"
