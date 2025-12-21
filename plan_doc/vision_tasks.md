# Vision & Multimodal Training

Karpathy-style Task pattern for training VLMs with DeepSeek-OCR algorithm.

## Data Format (Unified Conversation Format)

All tasks (vision and text) use the same conversation format for multimodal scalability:

```python
# Vision sample (FineVision format - PIL images directly)
{
    "messages": [
        {"role": "user", "content": "<image>\nOCR this document."},
        {"role": "assistant", "content": "Hello world"}
    ],
    "images": [PIL.Image],  # FineVision: PIL images directly
}

# Vision sample (path format - file path)
{
    "messages": [
        {"role": "user", "content": "<image>\nOCR this document."},
        {"role": "assistant", "content": "Hello world"}
    ],
    "image_path": "data/images/doc.png",  # path to file
}

# Text sample (same format, no media)
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
| `images` | Optional list of PIL.Image (FineVision datasets) |
| `image_path` | Optional path to image file (local datasets) |

This unified format:
- Supports both FineVision (`images`) and local (`image_path`) datasets
- Allows mixing vision + text in single TaskMixture
- ALL samples use `tokenizer.render_conversation()` - one tokenization path for all modalities

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
from tasks.finevision import FineVision
from tasks.smoltalk import SmolTalk

# Stage 1: OCR-focused (FineVision subsets)
train_ds = TaskMixture([
    # Document OCR
    FineVision("DoclingMatix", max_samples=500_000),
    FineVision("synthdog", max_samples=200_000),
    # Charts
    FineVision("SynthChartNet", max_samples=300_000),
    FineVision("Unichart", max_samples=200_000),
    # Math/LaTeX
    FineVision("latexformulas", max_samples=100_000),
])

# Stage 2: 70/20/10 mix (FineVision + existing text tasks)
train_ds = TaskMixture([
    # 70% OCR (FineVision)
    FineVision("DoclingMatix", max_samples=500_000),
    FineVision("SynthChartNet", max_samples=300_000),
    # 20% General Vision (FineVision)
    FineVision("vision_flan", max_samples=176_000),
    # 10% Text (existing tasks)
    SmolTalk(split="train"),
])

# One loader handles all modalities
train_loader = create_multimodal_loader(train_ds, tokenizer, B, T, base_size)
for inputs, targets, media in train_loader:
    # media["pixel_values"] is None for text-only batches
    loss = model(input_ids=inputs, targets=targets, pixel_values=media["pixel_values"])
```

Key insight: **modality is a property of the sample, not the loader.**

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

---

## FineVision: The VLM's FineWeb

**Discovery**: [HuggingFace FineVision](https://huggingface.co/datasets/HuggingFaceM4/FineVision) is FineWeb for vision! Unifies **200+ sources** into **185 subsets** with **24.3M samples**, **17.3M images**, and **9.5B answer tokens**.

### DeepSeek-OCR Training Data Reference

From [DeepSeek-OCR paper](https://arxiv.org/abs/2510.18234):

| Stage | Mix | Description |
|-------|-----|-------------|
| **Stage 1** | OCR 1.0+2.0 + LAION 100M | DeepEncoder training, 2 epochs |
| **Stage 2** | 70% OCR / 20% General Vision / 10% Text | Full model, 1 epoch |

**OCR 1.0 (~53M)**: 30M PDF pages, 3M Word docs, 20M scene text
**OCR 2.0 (~16M)**: 10M charts (pyecharts→HTML), 5M chemistry (SMILES→RDKit), 1M geometry

---

## FineVision → DeepSeek-OCR Mapping (Verified)

### 70% OCR Bucket

#### A. Document OCR (PDF/scanned/forms) - matches "PDF pages + Word docs"

| FineVision Subset | Size | Format | DeepSeek Equivalent |
|-------------------|------|--------|---------------------|
| **DoclingMatix** | 1.27M | PDF→Markdown (IBM Docling, DocTags) | PDF Fine annotation |
| **olmOCR-mix-0225-documents** | 229K | PDF→text (Allen AI) | PDF documents |
| **olmOCR-mix-0225-books** | 15.2K | Book pages→text | PDF books |
| **pdfvqa** | 8.59K | PDF→Q&A | Document understanding |
| **docvqa** | 10.2K | Document→Q&A | Document VQA |
| **invoices_receipts** | 3.01K | Invoice/receipt→structured | Forms/receipts |
| **sroie** | 33.6K | Receipt→key-value | Receipt OCR |
| **funsd** | 194 | Form→structure | Form understanding |

**Subtotal: ~2.1M document OCR samples**

#### B. Scene Text OCR - matches "LAION/Wukong + PaddleOCR"

| FineVision Subset | Size | Format | DeepSeek Equivalent |
|-------------------|------|--------|---------------------|
| **ocrvqa** | 166K | Scene→Q&A | Scene text understanding |
| **textvqa** | 21.9K | Scene→Q&A | Scene text VQA |
| **textcaps** | 21.9K | Scene→caption | Scene text description |
| **st_vqa** | 17.2K | Scene→Q&A | Scene text VQA |
| **ctw** | 23.8K | Chinese street text | Chinese scene OCR |
| **cocotext** | 16.2K | COCO images with text | Scene text |
| **iiit5k** | 1.99K | Word images | Word recognition |
| **screenqa** | 80.8K | Screenshot→Q&A | Screen/UI text |
| **screen2words** | 15.7K | Screenshot→description | Screen OCR |

**Subtotal: ~365K scene text samples**

#### C. OCR 2.0: Charts & Tables - matches "10M charts→HTML"

| FineVision Subset | Size | Format | Notes |
|-------------------|------|--------|-------|
| **SynthChartNet** | 500K (really 1.98M) | Chart→OTSL (5-token vocab) | Matplotlib/Seaborn/Pyecharts rendered |
| **Unichart** | 612K | Line chart→multi-turn Q&A | Quality rated (relevance/formatting) |
| **plotqa** | 157K | Plot→structured | Scientific plots |
| **dvqa** | 200K | Bar chart→Q&A | Bar chart understanding |
| **chartqa** | 18.3K | Chart→Q&A | General chart Q&A |
| **chart2text** | 27K | Chart→description | Chart summarization |
| **CoSyn_400k_chart** | 117K | Synthetic chart (Claude+GPT-4o) | Code-rendered charts |
| **CoSyn_400k_table** | 46.5K | Synthetic table (HTML/LaTeX) | Code-rendered tables |
| **hitab** | 2.5K | Hierarchical table→Q&A | Complex tables |
| **robut_wikisql** | 75K | Table→SQL | Table reasoning |
| **robut_wtq** | 38.2K | Table→Q&A | Table Q&A |

**Subtotal: ~1.8M chart/table samples**

**Note on OTSL**: Optimised Table Structure Language uses only 5 tokens (vs HTML's 28+), half the sequence length. Perfect for efficient table parsing.

#### D. OCR 2.0: Chemistry - matches "5M SMILES→RDKit"

| FineVision Subset | Size | Format | Notes |
|-------------------|------|--------|-------|
| **CoSyn_400k_chemical** | 8.94K | Chemical structure→SMILES-like | RDKit-rendered molecules |

**Gap**: DeepSeek has 5M chemistry samples. FineVision has only ~9K. **Must supplement with custom SMILES→RDKit pipeline or PubChem data.**

#### E. OCR 2.0: Geometry & Math - matches "1M geometry + LaTeX"

| FineVision Subset | Size | Format | Notes |
|-------------------|------|--------|-------|
| **latexformulas** | 552K | Formula image→LaTeX | Printed formulas |
| **latex_handwritten** | 39.6K | Handwritten→LaTeX | Handwritten math |
| **mathwriting-google** | 300K | Handwriting→text | Google handwriting |
| **hme100k** | 74.5K | Handwritten math→LaTeX | Math expressions |
| **geo170k(align)** | 35.3K | Geometry→alignment | Geometry pre-training |
| **geo170k(qa)** | 12.1K | Geometry→Q&A | Geometry reasoning |
| **geomverse** | 9.3K | Geometry→reasoning | Geometric reasoning |
| **geometry3k** | 9.72K | Geometry problem→solution | Problem solving |
| **unigeo** | 11.9K | Geometry→unified | Unified geometry |
| **intergps** | 1.28K | Geometry→GPS-style | Geometry parsing |
| **CoSyn_400k_math** | 66.7K | Synthetic math (LaTeX) | Code-rendered math |
| **CoSyn_400k_diagram** | 35K | Synthetic diagrams (Mermaid/Graphviz) | Code-rendered diagrams |

**Subtotal: ~1.15M geometry/math samples**

### 20% General Vision Bucket

| FineVision Subset | Size | Notes |
|-------------------|------|-------|
| **vision_flan(filtered)** | 176K | Diverse vision tasks |
| **LLaVA_Instruct_150K** | 158K | Instruction-following |
| **sharegpt4v(*)** | Multiple | GPT-4V annotations |
| **localized_narratives** | 200K | Localized image descriptions |
| **lvis_instruct4v** | 223K | Grounding/detection |

**Subtotal: ~750K+ general vision samples**

### 10% Text-Only Bucket

Reuse Karpathy's existing text datasets from `scripts/mid_train.py` instead of FineVision text subsets:

| Dataset | Size | Notes |
|---------|------|-------|
| **SmolTalk** | 460K | ShareGPT-style general conversations |
| **MMLU** | 100K | Multiple choice (ARC, MC_TEST, OBQA, RACE) |
| **GSM8K** | 8K | Grade school math + calculator tool use |
| **CustomJSON** | 2K | Identity conversations (2 epochs) |

**Subtotal: ~570K text samples**

**Rationale**: These datasets are battle-tested in Karpathy's nanoGPT/nanoChatGPT lineage and already integrated with our Task/TaskMixture system. No need to introduce new text data dependencies.

---

## Recommended Recipe for Nano DeepSeek-OCR

### Stage 1: DeepEncoder Training (OCR Focus)

```python
# ~4M samples for Stage 1
STAGE1_FINEVISION = {
    # Document OCR (~2.1M)
    "DoclingMatix": 1_270_000,      # PDF→Markdown (priority!)
    "synthdog": 500_000,            # Synthetic docs
    "olmOCR-mix-0225-documents": 229_000,
    "olmOCR-mix-0225-books": 15_200,
    "sroie": 33_600,

    # Scene Text (~365K)
    "ocrvqa": 166_000,
    "textvqa": 21_900,
    "textcaps": 21_900,
    "screenqa": 80_800,

    # Charts/Tables (OCR 2.0) (~1.1M)
    "SynthChartNet": 500_000,       # Charts→OTSL (scaled from 2M)
    "Unichart": 300_000,            # Sample from 612K
    "CoSyn_400k_chart": 117_000,
    "CoSyn_400k_table": 46_500,
    "plotqa": 157_000,

    # Math/Geometry (OCR 2.0) (~500K)
    "latexformulas": 300_000,       # Sample from 552K
    "hme100k": 74_500,
    "geo170k(align)": 35_300,
    "geo170k(qa)": 12_100,
    "CoSyn_400k_math": 66_700,
}
# Total: ~4.2M samples
```

### Stage 2: Full Model (70/20/10 Mix)

```python
# 70% OCR (from Stage 1 + more)
OCR_70 = STAGE1_FINEVISION  # ~4.2M

# 20% General Vision (~1.2M)
GENERAL_20 = {
    "vision_flan(filtered)": 176_000,
    "LLaVA_Instruct_150K": 158_000,
    "localized_narratives": 200_000,
    "lvis_instruct4v": 223_000,
    # Add from existing: LLaVA-CC3M
    "LLaVA-CC3M": 400_000,
}

# 10% Text (~570K) - Reuse existing tasks from mid_train.py
TEXT_10 = {
    "SmolTalk": 460_000,            # General conversations
    "MMLU": 100_000,                # Multiple choice Q&A
    "GSM8K": 8_000,                 # Math + tool use
    "CustomJSON": 2_000,            # Identity conversations (2 epochs)
}

# Total Stage 2: ~5.97M samples
# Ratio: 4.2M/1.2M/0.57M ≈ 70%/20%/10% ✓
```

### Loading FineVision Subsets

```python
from datasets import load_dataset, get_dataset_config_names

# List all 185 subsets
subsets = get_dataset_config_names("HuggingFaceM4/FineVision")

# Load specific subset (NO streaming - use HF native random access)
ds = load_dataset(
    "HuggingFaceM4/FineVision",
    name="DoclingMatix",
    split="train",
)

# FineVision unified format:
# {
#     "images": [PIL.Image],
#     "texts": [{"user": "...", "assistant": "..."}],
#     "source": "DoclingMatix",
#     "relevance_ratings": [...],  # optional quality ratings
# }

# Convert to our conversation format:
def finevision_to_nanochat(sample):
    texts = sample["texts"]
    return {
        "messages": [
            {"role": "user", "content": texts[0]["user"]},
            {"role": "assistant", "content": texts[0]["assistant"]},
        ],
        "images": sample["images"],  # PIL images directly
    }
```

### Data Loading Performance (Karpathy-style)

**Key insight**: For large datasets, follow Karpathy's FineWeb pattern - never cache in Python lists.

| Approach | Startup | RAM | Random Access |
|----------|---------|-----|---------------|
| `streaming=True` + list cache | ~26 min (18K samples) | O(n) - all in RAM | ✓ (after cache) |
| HF native (`streaming=False`) | 4.3s | O(1) - memory-mapped | ✓ (immediate) |

**How HuggingFace handles large data:**
1. Downloads to `~/.cache/huggingface/` (Arrow format)
2. Memory-maps files - data stays on disk
3. `ds[index]` loads only that sample on demand

**When to use streaming:**
- Single sequential pass (processing/filtering)
- Dataset too large for disk (100TB+)
- Quick preview without downloading

**When NOT to use streaming:**
- TaskMixture (requires random access for shuffling)
- Multiple epochs (re-downloads each time)
- Any use of `ds[index]`

**Correct implementation** (see `tasks/finevision.py`):
```python
class FineVision(Task):
    def __init__(self, subset_name: str, **kwargs):
        super().__init__(**kwargs)  # Passes start/stop/step to Task base class
        # HF native - memory-mapped Arrow files on disk
        self.ds = load_dataset(
            "HuggingFaceM4/FineVision",
            name=subset_name,
            split="train",
        )

    def get_example(self, index):
        return self.ds[index]  # Loaded from disk on-demand

# Usage:
FineVision("chartqa")                          # full dataset
FineVision("chartqa", stop=10000)              # first 10K samples
FineVision("chartqa", start=10000, stop=10100) # samples 10000-10099
```

**Comparison with Karpathy's FineWeb** (`nanochat/dataset.py`):
- FineWeb: Parquet files + `parquets_iter_batched()` generator
- FineVision: Arrow files + HF memory-mapping
- Both: Stream from disk, never cache in RAM

### Key Formats Discovered

| Subset | Output Format | Notes |
|--------|---------------|-------|
| SynthChartNet | **OTSL** (5 tokens) | Efficient table structure |
| DoclingMatix | **Markdown** (DocTags) | PDF→structured text |
| synthdog | **JSON** (gt_parse) | Donut-style ground truth |
| CoSyn_* | **Q&A pairs** | Claude code + GPT-4o Q&A |
| Unichart | **Multi-turn conversation** | With quality ratings |

### Gaps to Address

| DeepSeek Category | DeepSeek Size | FineVision Size | Gap |
|-------------------|---------------|-----------------|-----|
| PDF documents | 30M | ~2M | 93% gap |
| Scene text | 20M | ~365K | 98% gap |
| Charts | 10M | ~1.8M | 82% gap |
| Chemistry | 5M | ~9K | **99.8% gap** |
| Geometry | 1M | ~90K | 91% gap |

**Chemistry is critical gap**: Need custom SMILES→RDKit rendering pipeline.

### Quality Advantages of FineVision

1. **Decontaminated**: 1% benchmark leakage (vs 2-3% others)
2. **Quality ratings**: relevance, formatting, visual_dependency scores
3. **Unified format**: Already conversation-style
4. **GUI/Agentic bonus**: 4.3M samples not in DeepSeek-OCR

---

## Action Items

1. [ ] Download priority FineVision subsets: DoclingMatix, SynthChartNet, synthdog
2. [x] Create `tasks/finevision.py` wrapper for unified format
3. [ ] Build chemistry rendering pipeline (SMILES→RDKit→image)
4. [ ] Benchmark: FineVision vs current datasets vs combined
5. [ ] Consider OTSL format for chart parsing (more efficient than HTML)

---

## References

- [DeepSeek-OCR Paper](https://arxiv.org/abs/2510.18234)
- [FineVision Dataset](https://huggingface.co/datasets/HuggingFaceM4/FineVision)
- [FineVision Paper](https://arxiv.org/abs/2510.17269)
- [CoSyn Paper (ACL 2025)](https://arxiv.org/abs/2502.14846)
- [OTSL Paper](https://arxiv.org/abs/2305.03393)
- [SynthDoG/Donut](https://github.com/clovaai/donut)
- [IBM Docling](https://github.com/docling-project/docling)
