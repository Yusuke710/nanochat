# Vision Dataloader

Karpathy-style vision dataloader extending nanochat's Task pattern.

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
- Variable image token count (73-900+ per image)

## Core Implementation

```python
# vision_dataloader.py
"""
Vision dataloader extending nanochat's sft_data_generator pattern.

Key differences from nanochat SFT:
1. Samples have optional image_path
2. <image> token expands to N copies (based on resolution)
3. Returns pixel_values alongside inputs/targets
4. Image token positions are masked in targets
"""

import torch
from PIL import Image

IMAGE_TOKEN = "<image>"

def vision_data_generator(dataset, batch_size, tokenizer, processor, device="cuda",
                          ddp_rank=0, ddp_world_size=1):
    """
    Args:
        dataset: Task or TaskMixture returning {"text", "response", "image_path"?}
        batch_size: samples per batch
        tokenizer: nanochat tokenizer (with <image> token added)
        processor: ImageProcessor for count_tokens() and process_image()

    Yields:
        inputs: [B, T] token IDs
        targets: [B, T] shifted IDs with image positions = -1
        pixel_values: [B, C, H, W] or None for text-only batches
    """
    IMAGE_TOKEN_ID = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    PAD_TOKEN_ID = tokenizer.encode_special("<|assistant_end|>")

    def process_sample(sample):
        """Convert sample dict → (token_ids, pixel_values)"""
        text = f"{sample['text']}{sample['response']}<|assistant_end|>"

        # Text-only sample
        if "image_path" not in sample:
            ids = tokenizer.encode(text, prepend="<|bos|>")
            return ids, None

        # Vision sample
        image = Image.open(sample["image_path"]).convert("RGB")
        n_tokens = processor.count_tokens()

        ids = tokenizer.encode(text, prepend="<|bos|>")

        # Expand <image> → N copies
        expanded = []
        for tok in ids:
            if tok == IMAGE_TOKEN_ID:
                expanded.extend([IMAGE_TOKEN_ID] * n_tokens)
            else:
                expanded.append(tok)

        pixel_values = processor.process_image(image)
        return expanded, pixel_values

    def collate(batch):
        """Pad batch and create targets with image masking."""
        max_len = max(len(ids) - 1 for ids, _ in batch)

        inputs = torch.full((len(batch), max_len), PAD_TOKEN_ID, dtype=torch.long)
        targets = torch.full((len(batch), max_len), -1, dtype=torch.long)
        pixel_values_list = []

        for i, (ids, pv) in enumerate(batch):
            n = len(ids) - 1
            ids_tensor = torch.tensor(ids, dtype=torch.long)

            inputs[i, :n] = ids_tensor[:-1]
            row_targets = ids_tensor[1:n+1].clone()

            # Mask image token positions (don't predict <image> tokens)
            row_targets[ids_tensor[:n] == IMAGE_TOKEN_ID] = -1
            targets[i, :n] = row_targets

            if pv is not None:
                pixel_values_list.append(pv)

        pixel_values = torch.stack(pixel_values_list) if pixel_values_list else None

        return (
            inputs.to(device),
            targets.to(device),
            pixel_values.to(device) if pixel_values is not None else None,
        )

    # Main loop (infinite epochs)
    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            sample = dataset[i]
            ids, pv = process_sample(sample)
            batch.append((ids, pv))

            if len(batch) == batch_size:
                yield collate(batch)
                batch = []
```

## ImageProcessor (from image_process.py)

```python
import math
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps

class ImageProcessor:
    def __init__(self, base_size=1024, patch_size=16, downsample_ratio=4):
        self.base_size = base_size
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.transform = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])

    def process_image(self, image: Image.Image) -> torch.Tensor:
        """Pad image to base_size and normalize."""
        global_view = ImageOps.pad(image, (self.base_size, self.base_size), color=(127, 127, 127))
        return self.transform(global_view)

    def count_tokens(self) -> int:
        """Token count for current resolution."""
        nq = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
        return (nq + 1) * nq + 1  # 273 for base_size=1024
```

## Training Script Usage

```python
# scripts/vis_tok_train.py
from tasks.common import TaskMixture
from vision_dataloader import vision_data_generator
from image_process import ImageProcessor

# Stage 1: Vision only
train_ds = TaskMixture([
    DocBank(), OlmOCR(), LLaVACC3M(), PlotQA(),
    ChartQA(), FigureQA(), PubTables1M(), LaTeXFormulas(),
])

processor = ImageProcessor(base_size=1024)
train_loader = vision_data_generator(
    train_ds,
    batch_size=device_batch_size,
    tokenizer=tokenizer,
    processor=processor,
    device=device,
    ddp_rank=ddp_rank,
    ddp_world_size=ddp_world_size,
)

# Training loop
for step in range(num_iterations):
    inputs, targets, pixel_values = next(train_loader)
    loss = model(inputs, targets, pixel_values=pixel_values)
    loss.backward()
    # ... optimizer step
```

## Stage 2: Mixed Vision + Text

```python
# TaskMixture shuffles automatically
# Text samples (no image_path) return pixel_values=None
train_ds = TaskMixture([
    # Vision (~90%)
    DocBank(), OlmOCR(), LLaVACC3M(), PlotQA(),
    ChartQA(), FigureQA(), PubTables1M(), LaTeXFormulas(),
    # Text (~10%)
    SmolTalk(stop=50000),
])
```

## Data Flow

```
Task.get_example(i)
    ↓
{"text": "<image>\nOCR this.", "response": "Hello", "image_path": "doc.png"}
    ↓
process_sample()
    ├── tokenizer.encode() → [BOS, IMG, ..., text_tokens..., EOS]
    ├── expand <image> → [BOS, IMG, IMG, IMG, ... (×273), ..., EOS]
    └── processor.process_image() → [3, 1024, 1024]
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

```python
# In collate():
row_targets[ids_tensor[:n] == IMAGE_TOKEN_ID] = -1

# Result for "<image>\nOCR this." with 3 image tokens:
# inputs:  [BOS, IMG, IMG, IMG, \n,  O,   C,   R, ...]
# targets: [-1,  -1,  -1,  -1,  \n,  O,   C,   R, ...]  (IMG positions masked)
```
