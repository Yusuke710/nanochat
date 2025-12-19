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
- Uses `tokenizer.render_conversation()` for tokenization

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
Sequence: [IMG×273, \n, O, C, R, ..., response_tokens]
Targets:  [-1×273,  \n, O, C, R, ..., response_tokens]  ← train on ALL text
```

- Train on ALL text tokens (not just response)
- Mask only image token positions
- Uses `tokenizer.render_conversation()` but ignores the mask

```python
# tokenizer.render_conversation() returns (ids, mask)
# We ignore the mask and train on all tokens
ids, _ = tokenizer.render_conversation(example)
targets = ids[1:]  # next-token prediction
for start, end in image_positions:
    targets[start:end-1] = -1  # ignore image tokens only
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
for inputs, targets, pixel_values in multimodal_data_generator(train_ds, tokenizer, ...):
    # pixel_values is None for text-only samples
    loss = model(input_ids=inputs, targets=targets, pixel_values=pixel_values)
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

The current `vis_mid_train.py` uses separate loaders with `text_ratio` mixing:

```python
# Vision loader
vision_loader = vision_data_generator(vision_ds, tokenizer, batch_size, base_size)

# Text loader (FineWeb parquet - to be replaced with SFT TaskMixture)
text_loader = tokenizing_distributed_data_loader(B=batch_size, T=seq_len, split="train")

# Mixing at step level
if random.random() < text_ratio:
    inputs, targets = next(text_loader)
    pixel_values = None
else:
    inputs, targets, pixel_values = next(vision_loader)
```

This will be migrated to the unified approach above.

## Migration Path

1. **Current**: Separate loaders, different formats (see "Current Implementation" above)
2. **Step 1**: Update vision tasks to return conversation format
3. **Step 2**: Create `multimodal_data_generator` that handles conversation format + optional media
4. **Step 3**: Unify into single TaskMixture in `vis_mid_train.py`

## Files

```
nanochat/
├── vision_dataloader.py    # create_vision_loader, vision_data_generator
├── dataloader.py           # tokenizing_distributed_data_loader (legacy text)
tasks/
├── overfit_samples.py      # Vision task for overfit testing
├── common.py               # Task, TaskMixture base classes
├── smoltalk.py             # Text SFT task
├── gsm8k.py                # Text SFT task
├── mmlu.py                 # Text SFT task
scripts/
├── vis_tok_train.py        # Stage 1: Vision encoder training
├── vis_mid_train.py        # Stage 2: Mixed vision + text training
├── mid_train.py            # Text-only mid-training (reference)
```
