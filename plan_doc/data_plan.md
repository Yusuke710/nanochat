# Data Plan: OCR Dataset Preparation for nanochat Training

## Overview

This document describes how to transform 9 selected Hugging Face datasets into the nanochat-compatible format for DeepSeek-OCR **mid-training** (not SFT).

**Key Design Decision**: Vision mid-training uses **pre-training loss** (supervise all text tokens), matching nanochat's `mid_train.py` pattern and the DeepSeek-OCR paper.

---

## Target Formats

### Format 1: Vision Pre-training (OCR, documents, charts, math)

Used for most vision data. Simple format **without conversation markers**, matching DeepSeek-OCR paper:

```python
{
    "text": "<image>\n{output_text}",    # <image> placeholder + target text
    "image_path": str                     # Path to image file
}
```

Tokenized via `render_vision_pretraining()` → `<|bos|>[image_tokens × N]\n{output_text}<|assistant_end|>`

**Note:** `<|assistant_end|>` is appended during data loading to teach the model when to stop (consistent with nanochat inference).

**Loss**: Supervise ALL text tokens (pre-training style)

---

## Image Token Expansion

**Important**: The `<image>` placeholder in text is NOT a single token. It expands to **hundreds of image tokens** based on image size:

```python
# From DeepSeek-OCR image_process.py (lines 424-435):
# For base_size=1024, image_size=640, patch_size=16, downsample_ratio=4:
num_queries_base = ceil(1024 / 16 / 4)  # = 16

# Global view: (16+1) * 16 + 1 = 273 tokens
tokenized_image = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base
tokenized_image += [image_token_id]

# Plus local crops if image > 640px (can add 100+ more tokens per crop)
```

The actual number of image tokens depends on:
- Image dimensions (aspect ratio determines number of crops)
- `MIN_CROPS` to `MAX_CROPS` settings (typically 1-6 crops)
- Typical range: **256-1000+ tokens per image**

---

## Loss Strategy: Pre-training (NOT Selective Supervision)

**Key insight**: Vision mid-training uses **pre-training loss** - supervise ALL text tokens, not just responses. This matches nanochat's `mid_train.py` and the DeepSeek-OCR approach.

```python
# What gets masked (loss NOT computed):
# 1. Image placeholder tokens only → targets[image_positions] = -1

# What gets trained (loss computed):
# 1. ALL text tokens (including prompts, markers, responses)
# 2. BOS token prediction is supervised
```

**This differs from SFT** where only assistant responses are supervised. Mid-training is about adding capability, not instruction following.

```python
# In collate_multimodal_batch:
targets = ids[1:]  # Standard next-token prediction

# Only mask image positions
for start, end in image_positions:
    targets[start:end-1] = -1  # ignore_index
```

---

## Selected Datasets

| Category | Dataset | HuggingFace ID | Size | Format Type |
|----------|---------|----------------|------|-------------|
| Document Layout | DocBank | `liminghao1630/DocBank` | 500k | Vision (text/response) |
| PDF OCR | olmOCR-mix | `allenai/olmOCR-mix-1025` | 268k | Vision (text/response) |
| General Vision | LLaVA-CC3M | `liuhaotian/LLaVA-CC3M-Pretrain-595K` | 595k | Vision (text/response) |
| Charts | PlotQA | `achang/plot_qa` | 224k | Vision (text/response) |
| Charts | ChartQA | `HuggingFaceM4/ChartQA` | 32k | Vision (text/response) |
| Charts | FigureQA | `vikhyatk/figureqa` | 100k+ | Vision (text/response) |
| Tables | PubTables-1M | `bsmock/pubtables-1m` | 947k | Vision (text/response) |
| Math (Printed) | LaTeX-Formulas | `OleehyO/latex-formulas` | 550k | Vision (text/response) |
| Math (Handwritten) | MathWriting | `deepcopy/MathWriting-human` | 230k | Vision (text/response) |

---

## Dataset Transformations

### 1. DocBank (`liminghao1630/DocBank`)

**Source Format:**
```python
{
    "image": PIL.Image,
    "words": ["token1", "token2", ...],
    "bboxes": [[x0, y0, x1, y1], ...],
    "structures": ["paragraph", "title", ...]  # 12 label types
}
```

**Transformation Strategy:**
- Save images to disk, store path
- Concatenate all tokens into reading-order text
- Use structure labels to add formatting (e.g., `## {title}`, `**{section}**`)
- Format: `<image>\n{output_text}` (vision pre-training style)

**Target Format (Vision Pre-training):**
```python
{
    "text": "<image>\n## Title\n\nParagraph text here...\n\n**Section Header**\n...",
    "image_path": "data/docbank/train/00001.png"
}
```

**Task Class:** `tasks/docbank.py`

---

### 2. olmOCR-mix (`allenai/olmOCR-mix-1025`)

**Source Format:**
```python
{
    "pdf": bytes,           # Single-page PDF
    "text": str,            # Markdown transcription (from GPT-4o)
    "metadata": {...}       # Source info
}
```

**Transformation Strategy:**
- Render PDF page to image using `pdf2image` or `PyMuPDF`
- Save rendered image to disk
- Use markdown text as ground truth
- Format: `<image>\n{markdown_text}` (vision pre-training style)

**Target Format (Vision Pre-training):**
```python
{
    "text": "<image>\n# Heading\n\nParagraph with **bold** and *italic*...",
    "image_path": "data/olmocr/train/doc_00001.png"
}
```

**Task Class:** `tasks/olmocr.py`

---

### 3. LLaVA-CC3M-Pretrain-595K (`liuhaotian/LLaVA-CC3M-Pretrain-595K`)

**Source Format:**
```python
# chat.json format
{
    "id": str,
    "image": "image_filename.jpg",
    "conversations": [
        {"from": "human", "value": "<image>\nDescribe this image."},
        {"from": "gpt", "value": "A cat sitting on a couch..."}
    ]
}
```

**Transformation Strategy:**
- Convert to plain prompt format (same as all other vision data)
- Extract human message as prompt, gpt message as response
- Use vision pre-training format (NO conversation markers)

**Target Format (Vision Pre-training - plain prompt):**
```python
{
    "text": "<image>\nDescribe this image.",
    "response": "A cat sitting on a couch...",
    "image_path": "data/llava_cc3m/images/00001.jpg"
}
```

**Task Class:** `tasks/llava_cc3m.py`

---

### 4. PlotQA (`achang/plot_qa`)

**Source Format:**
```python
{
    "image": PIL.Image,
    "text": str  # Answer/target text
}
```

**Transformation Strategy:**
- Save images to disk
- Text is the answer; add question prompts
- Use varied prompts: "Describe this chart", "What does this plot show?"

**Target Format:**
```python
{
    "text": "<image>\nDescribe this chart and its data.",
    "response": "This bar chart shows sales data from 2010-2020...",
    "image_path": "data/plotqa/train/00001.png"
}
```

**Task Class:** `tasks/plotqa.py`

---

### 5. ChartQA (`HuggingFaceM4/ChartQA`)

**Source Format:**
```python
{
    "image": PIL.Image,
    "query": str,           # Question about the chart
    "label": str,           # Answer
    "type": str             # "human" or "machine" generated
}
```

**Transformation Strategy:**
- Save images to disk
- Prompt includes the question
- Response is the answer

**Target Format:**
```python
{
    "text": "<image>\nQuestion: What is the highest value shown in this chart?",
    "response": "The highest value is 42.5 in 2019.",
    "image_path": "data/chartqa/train/00001.png"
}
```

**Task Class:** `tasks/chartqa.py`

---

### 6. FigureQA (`vikhyatk/figureqa`)

**Source Format:**
```python
{
    "image": PIL.Image,
    "question": str,        # Yes/no question about the figure
    "answer": bool          # True/False
}
```

**Transformation Strategy:**
- Save images to disk
- Binary Q/A about synthetic figures
- Convert boolean to "Yes"/"No" string

**Target Format:**
```python
{
    "text": "<image>\nQuestion: Is the blue line above the red line?",
    "response": "Yes",
    "image_path": "data/figureqa/train/00001.png"
}
```

**Task Class:** `tasks/figureqa.py`

---

### 7. PubTables-1M (`bsmock/pubtables-1m`)

**Source Format:**
```
# Separate files:
- images/*.jpg           # Cropped table images
- annotations/*.xml      # PASCAL VOC format with cell bboxes
- words/*.json           # Text content per cell
```

**Transformation Strategy:**
- Images are already on disk (download as tar.gz)
- Combine cell bboxes with text content
- Generate HTML table representation
- Prompt: "Convert this table to HTML"

**Target Format:**
```python
{
    "text": "<image>\nConvert this table to HTML.",
    "response": "<table><tr><th>Header1</th><th>Header2</th></tr><tr><td>Cell1</td><td>Cell2</td></tr></table>",
    "image_path": "data/pubtables/structure/images/PMC123456_table_0.jpg"
}
```

**Task Class:** `tasks/pubtables.py`

---

### 8. LaTeX-Formulas (`OleehyO/latex-formulas`)

**Source Format (cleaned_formulas):**
```python
{
    "image": PIL.Image,
    "latex_formula": str    # LaTeX code
}
```

**Transformation Strategy:**
- Save images to disk
- Direct image -> LaTeX conversion
- Prompt: "Convert this formula to LaTeX"

**Target Format:**
```python
{
    "text": "<image>\nConvert this formula to LaTeX.",
    "response": "\\frac{1}{2} \\sum_{i=1}^{n} x_i^2",
    "image_path": "data/latex_formulas/train/00001.png"
}
```

**Task Class:** `tasks/latex_formulas.py`

---

### 9. MathWriting (`deepcopy/MathWriting-human`)

**Source Format:**
```python
{
    "image": PIL.Image,     # Handwritten expression image
    "latex": str            # LaTeX representation
}
```

**Transformation Strategy:**
- Save images to disk
- Similar to LaTeX-Formulas but for handwritten
- Prompt emphasizes handwritten recognition

**Target Format:**
```python
{
    "text": "<image>\nRecognize this handwritten formula and convert to LaTeX.",
    "response": "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}",
    "image_path": "data/mathwriting/train/00001.png"
}
```

**Task Class:** `tasks/mathwriting.py`

---

## Task Class Template

### Vision Pre-training Task (text format - OCR, docs, charts, math)

```python
# tasks/example_ocr.py
from datasets import load_dataset
from tasks.common import Task
import os

class ExampleOCR(Task):
    """Example OCR dataset task - vision pre-training format."""

    def __init__(self, split="train", data_dir="data/example", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "val", "test"]
        self.ds = load_dataset("huggingface/dataset-id", split=split)
        self.data_dir = data_dir

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]

        # Image path (pre-saved or construct from index)
        image_path = os.path.join(self.data_dir, f"{index:08d}.png")

        # Vision pre-training format: <image>\n{output_text}
        # NO conversation markers, simple and direct
        return {
            "text": f"<image>\n{row['text']}",
            "image_path": image_path,
        }
```

### LLaVA-CC3M Task (plain prompt format - same as all vision data)

```python
# tasks/llava_cc3m.py
from datasets import load_dataset
from tasks.common import Task
import os

class LLaVACC3M(Task):
    """LLaVA CC3M dataset - uses plain prompt format like all vision data."""

    def __init__(self, split="train", data_dir="data/llava_cc3m", **kwargs):
        super().__init__(**kwargs)
        self.ds = load_dataset("liuhaotian/LLaVA-CC3M-Pretrain-595K", split=split)
        self.data_dir = data_dir

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        conversations = row["conversations"]

        # Extract prompt (human) and response (gpt) - convert to plain format
        prompt = ""
        response = ""
        for conv in conversations:
            if conv["from"] == "human":
                prompt = conv["value"]  # e.g., "<image>\nDescribe this image."
            else:
                response = conv["value"]

        image_path = os.path.join(self.data_dir, "images", row["image"])

        # Plain prompt format (same as all other vision data)
        return {
            "text": prompt,
            "response": response,
            "image_path": image_path,
        }
```

---

## Dataloader: Why SFT-Style (Not Token Packing)

### nanochat Has Two Batch Strategies

| Script | Strategy | Description |
|--------|----------|-------------|
| `mid_train.py` | Token packing | Accumulate tokens in buffer, yield dense `(B, T)`, train on ALL tokens |
| `chat_sft.py` | Per-conversation | Each sample is a unit, pad shorter sequences, use mask for supervision |

### Why Vision Needs SFT-Style

**Token packing doesn't work for vision** because:
1. Image samples can't be split mid-conversation (image + response = atomic unit)
2. `images_seq_mask` must align with token positions
3. `pixel_values` tensor must correspond to exactly the `<image>` tokens in that batch

```
Token packing (mid_train.py) - CAN'T work for vision:
┌─────────────────────────────────────────────────────┐
│ [conv1_tokens...][conv2_tokens...][conv3_tok...     │  <- where do images go?
└─────────────────────────────────────────────────────┘

Per-sample (chat_sft.py) - Works for vision:
┌─────────────────────────────────────────────────────┐
│ Sample 1: [<image>_N_tokens][prompt][response][PAD] │  pixel_values[0]
│ Sample 2: [<image>_M_tokens][prompt][response][PAD] │  pixel_values[1]
└─────────────────────────────────────────────────────┘
```

### Vision Data Generator (from training_plan.md Section 5.1)

```python
from nano_dpsk_ocr.data.sample_adapter import normalize_sample, validate_sample

def vision_data_generator(dataset, batch_size):
    """Per-sample batching for vision. Uses PRE-TRAINING LOSS (supervise all text)."""

    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(b["ids"]) for b in batch) - 1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)
        images_seq_mask = torch.zeros((nrows, ncols), dtype=torch.bool)

        for i, sample in enumerate(batch):
            ids = sample["ids"]
            img_pos = sample.get("image_positions", [])
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n-1] = ids_tensor[:-1]

            # PRE-TRAINING LOSS: supervise ALL text tokens
            targets[i, :n-1] = ids_tensor[1:]

            # Only mask image placeholder positions
            for start, end in img_pos:
                targets[i, start:end-1] = -1  # ignore_index for image tokens
                images_seq_mask[i, start:end-1] = True

        pixel_values_list = [b["pixel_values"] for b in batch if b["pixel_values"] is not None]
        pixel_values = torch.stack(pixel_values_list) if pixel_values_list else None

        return inputs.to(device), targets.to(device), pixel_values, images_seq_mask.to(device)

    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            sample = dataset[i]

            # === ADAPTER: Normalize format at boundary ===
            sample = normalize_sample(sample)  # {messages} -> {text, response}
            validate_sample(sample)            # Fail fast on malformed data

            # Build full text with <|assistant_end|> stop token
            full_text = f"{sample['text']}{sample.get('response', '')}<|assistant_end|>"

            if "image_path" in sample:
                # Process image
                image = Image.open(sample["image_path"])
                vision_out = processor.process_images([image])
                num_tokens = vision_out["num_tokens_per_image"]

                # Plain prompt tokenization (NO conversation markers)
                ids, img_pos = tokenizer.render_vision_pretraining(
                    full_text, image_token_counts=num_tokens
                )

                batch.append({
                    "ids": ids, "image_positions": img_pos,
                    "pixel_values": vision_out["pixel_values"],
                })
            else:
                # Text-only: plain tokenization
                bos = tokenizer.get_bos_token_id()
                ids = [bos] + tokenizer.encode(full_text)
                batch.append({
                    "ids": ids, "image_positions": [],
                    "pixel_values": None,
                })

            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []
```

### Training Loop Integration

```python
# Training loop (same structure as chat_sft.py, extended for vision)
for step in range(num_iterations):
    inputs, targets, pixel_values, images_seq_mask = next(train_loader)

    with autocast_ctx:
        # Vision embedding merge happens inside model.forward()
        loss = model(inputs, targets, pixel_values, images_seq_mask)

    loss = loss / grad_accum_steps
    loss.backward()
    # ... optimizer step ...
```

### Compatibility Analysis

| Dataset | Format | Has `image_path` | Masking | Compatible |
|---------|--------|------------------|---------|------------|
| DocBank | text/response | ✅ | image tokens only | ✅ |
| olmOCR-mix | text/response | ✅ | image tokens only | ✅ |
| LLaVA-CC3M | text/response | ✅ | image tokens only | ✅ |
| PlotQA | text/response | ✅ | image tokens only | ✅ |
| ChartQA | text/response | ✅ | image tokens only | ✅ |
| FigureQA | text/response | ✅ | image tokens only | ✅ |
| PubTables-1M | text/response | ✅ | image tokens only | ✅ |
| LaTeX-Formulas | text/response | ✅ | image tokens only | ✅ |
| MathWriting | text/response | ✅ | image tokens only | ✅ |

All datasets use `image_path` (string path), not PIL.Image directly. This is more memory-efficient.

---

## Training Mixture Configuration

### Stage 1: DeepEncoder Training (Vision Only, LLM Frozen)

All vision datasets are used in Stage 1, including LLaVA-CC3M (it has images, so gradients flow through the vision encoder even though the LLM is frozen).

```python
from tasks.common import TaskMixture

stage1_ds = TaskMixture([
    # Document OCR (high weight - core task)
    DocBank(split="train"),
    DocBank(split="train"),  # 2x weight
    OlmOCR(split="train"),
    OlmOCR(split="train"),   # 2x weight

    # General vision understanding (plain prompt format)
    LLaVACC3M(split="train"),

    # Charts and figures
    PlotQA(split="train"),
    ChartQA(split="train"),
    FigureQA(split="train"),

    # Tables
    PubTables(split="train"),

    # Math formulas
    LaTeXFormulas(split="train"),
    MathWriting(split="train"),
])
```

**Note**: Text-only tasks (SmolTalk, GSM8K) are NOT included in Stage 1 to focus training on vision-text alignment. Stage 2 adds text-only data to preserve language capabilities.

### Stage 2: Full Fine-tuning (Vision 90% + Text 10%)

```python
stage2_ds = TaskMixture([
    # === Vision tasks (90% total) ===

    # Document OCR (high weight)
    *[DocBank(split="train") for _ in range(2)],
    *[OlmOCR(split="train") for _ in range(2)],

    # General vision understanding (plain prompt format)
    LLaVACC3M(split="train"),

    # Charts and figures
    PlotQA(split="train"),
    ChartQA(split="train"),
    FigureQA(split="train"),

    # Tables
    PubTables(split="train"),

    # Math formulas
    LaTeXFormulas(split="train"),
    MathWriting(split="train"),

    # === Text-only tasks (10% - to preserve text ability) ===
    SmolTalk(split="train", stop=50000),  # Subset to ~10% of total
])
```

**Note**: All vision datasets (including LLaVA-CC3M) use the same plain prompt format `{text, response, image_path}`.

---

## Data Preprocessing Pipeline

### Step 1: Download & Extract

```bash
# Create data directory
mkdir -p data/{docbank,olmocr,llava_cc3m,plotqa,chartqa,figureqa,pubtables,latex_formulas,mathwriting}

# Download via HuggingFace datasets (cached automatically)
python scripts/download_datasets.py
```

### Step 2: Save Images to Disk (for datasets that return PIL.Image)

```python
# scripts/preprocess_images.py
from datasets import load_dataset
from PIL import Image
import os

def save_images(ds, output_dir, split="train"):
    os.makedirs(output_dir, exist_ok=True)
    for i, row in enumerate(ds):
        img = row["image"]
        if isinstance(img, Image.Image):
            img.save(os.path.join(output_dir, f"{i:08d}.png"))
        if i % 10000 == 0:
            print(f"Saved {i} images...")

# Example for LaTeX-Formulas
ds = load_dataset("OleehyO/latex-formulas", "cleaned_formulas", split="train")
save_images(ds, "data/latex_formulas/train")
```

### Step 3: Create Index Files (Optional - for fast loading)

```python
# Save metadata as JSONL for fast task loading
import json

def create_index(ds, output_file, image_dir):
    with open(output_file, "w") as f:
        for i, row in enumerate(ds):
            entry = {
                "image_path": os.path.join(image_dir, f"{i:08d}.png"),
                "text": row["latex_formula"],  # or appropriate field
            }
            f.write(json.dumps(entry) + "\n")
```

---

## File Structure

```
nano-deepseek-ocr/
├── tasks/
│   ├── __init__.py
│   ├── common.py          # Base Task, TaskMixture (from nanochat)
│   ├── docbank.py
│   ├── olmocr.py
│   ├── llava_cc3m.py      # Plain prompt format (same as all vision tasks)
│   ├── plotqa.py
│   ├── chartqa.py
│   ├── figureqa.py
│   ├── pubtables.py
│   ├── latex_formulas.py
│   └── mathwriting.py
├── scripts/
│   ├── download_datasets.py
│   ├── preprocess_images.py
│   └── validate_datasets.py
└── data/
    ├── docbank/
    │   └── train/
    ├── olmocr/
    │   └── train/
    ├── llava_cc3m/
    │   └── images/
    ├── plotqa/
    │   └── train/
    ├── chartqa/
    │   └── train/
    ├── figureqa/
    │   └── train/
    ├── pubtables/
    │   └── structure/images/
    ├── latex_formulas/
    │   └── train/
    └── mathwriting/
        └── train/
```

---

## Validation Checklist

- [ ] Each task class returns `{text, image_path}` format (plain prompt)
- [ ] `text` field starts with `<image>\n`
- [ ] All `image_path` values are valid file paths (strings)
- [ ] Images are saved as PNG/JPG files on disk
- [ ] TaskMixture correctly shuffles all datasets
- [ ] Dataloader uses `render_vision_pretraining()` for tokenization
- [ ] Pre-training loss used: supervise ALL text tokens
- [ ] Only image placeholder positions are masked (not prompts)
- [ ] Memory-efficient: images loaded on-demand, not stored in memory

---

## Key Design Decisions

### Plain Prompt Tokenization

| Method | Use Case | Format | Loss |
|--------|----------|--------|------|
| `render_vision_pretraining()` | All vision data | `<\|bos\|><image>\ntext<\|assistant_end\|>` | All text tokens |

### Component Architecture

| Component | Responsibility |
|-----------|----------------|
| `render_vision_pretraining` | Simple tokenization for vision pre-training (NO markers) |
| `VisionProcessor.process_images` | Image → tensors (no tokenization) |
| `collate_multimodal_batch` | Combine outputs + pre-training loss |

### Key Points

1. **Image handling**: Always use `image_path` (string), never PIL.Image in dataset output
2. **Single data format**: `{"text": "<image>\n...", "image_path": str}` (plain prompt, no conversation markers)
3. **Pre-training loss**: Supervise ALL text tokens (not selective supervision)
4. **Only image positions masked**: `targets[image_positions] = -1`
5. **Image token expansion**: `<image>` → N tokens (256-1000+) based on actual image size
6. **SFT-style batching**: Per-sample (not token packing) because vision samples are atomic units

### Why Pre-training Loss (Not SFT)?

This is **mid-training** - adding vision capability to an existing LLM:
- We want the model to learn raw vision→text capability
- Not instruction following (that comes later with SFT if needed)
- Matches DeepSeek-OCR paper approach
- Simple format: `<image>\ntext` (no conversation markers needed)
