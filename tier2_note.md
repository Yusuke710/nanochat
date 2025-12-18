# Tier 2: Vision Tasks with olmOCR Dataset

## Overview
Adding document OCR task using the `allenai/olmOCR-mix-1025` dataset from HuggingFace.

## Dataset: olmOCR-mix-1025
- Source: https://huggingface.co/datasets/allenai/olmOCR-mix-1025
- ~270K PDF pages OCR'd by GPT-4.1
- High-quality ground truth with natural reading order
- Contains: documents, books, transcripts, national archives

## Data Preparation
- Script: `scripts/prepare_olmocr.py`
- Output: `data/olmocr/`
  - `train.json`: 300 samples
  - `val.json`: 10 samples
  - `images/`: JPEG images converted from PDFs at 150 DPI

### Content Type Distribution
```
Train (300 samples):
  - Tables: 26 (8.7%)
  - Diagrams: 14 (4.7%)
  - Documents: 260 (86.6%)

Val (10 samples):
  - Tables: 0
  - Diagrams: 2
  - Documents: 8
```

## Sample Format (Karpathy style)

### Updated: Type-Specific Prompts (2024-12-18)
Based on DeepSeek-OCR paper findings, we now use different prompts for different content types:

```json
// For regular documents
{
  "id": "olm_00000",
  "type": "document",
  "prompt": "<image>\nOCR this document.",
  "answer": "GEORGE MASON UNIVERSITY..."
}

// For tables
{
  "id": "olm_00002",
  "type": "table",
  "prompt": "<image>\nConvert this table to HTML.",
  "answer": "<table>..."
}

// For diagrams
{
  "id": "olm_00010",
  "type": "diagram",
  "prompt": "<image>\nDescribe this diagram.",
  "answer": "This diagram shows..."
}
```

**Rationale**: DeepSeek-OCR paper uses different prompts for different content types:
- Documents: `<image>\n<|grounding|>Convert the document to markdown.`
- Tables: Chart Deep Parsing mode (outputs HTML)
- Plain text: `<image>\nFree OCR.`

Source: https://arxiv.org/abs/2510.18234

## Training Progress

### Run 1: Initial Test (2024-12-18)
- Config: `batch_size=2, steps=300, lr=5e-5`
- Results:
  - Train loss: 5.80 → 3.29
  - Val loss: 5.79 → 5.25
  - Time: 8.4 min
- Status: **Incomplete** - need more steps

### Run 2: Longer Training (2024-12-18)
- Script: `vis_tok_train.py` (pulled from tier1 branch)
- Config: `batch_size=4, steps=2000, lr=5e-5, seq_len=4096`
- Results:
  - Train loss: 6.4 → 0.44
  - Val loss: 4.07 → 8.15 (min: 3.94 at step 200)
  - Time: 18.68 min
- Checkpoint: `checkpoints/step_2000.pt`
- Status: **Overfitting** - val loss increased while train loss decreased

### Evaluation Results (Run 2)
- Generated HTML report: `data/inference_results.html`
- **Problem**: Model outputs HTML table format for ALL inputs
- **Root cause**:
  1. Single prompt used for all content types
  2. Model memorized table format from training data
  3. Overfitting (2000 steps on 300 samples)

### Run 3: With Type-Specific Prompts (TODO)
- Updated `prepare_olmocr.py` to use type-specific prompts
- Updated `data/olmocr/train.json` and `val.json`
- Need to retrain with:
  - Earlier stopping (step ~200 had best val loss)
  - Type-specific prompts to differentiate content

## Batch Size Testing
| batch_size | seq_len | Result |
|------------|---------|--------|
| 10 | 4096 | OOM |
| 8 | 4096 | OOM (sometimes works initially, fails on longer sequences) |
| 6 | 4096 | OOM (fails after ~16 steps) |
| 4 | 4096 | **Works** - stable training |

## Notes
- GPU: NVIDIA A100-SXM4-80GB
- Max stable batch_size=4 with seq_len=4096 (variable length sequences cause OOM)
- Best checkpoint so far: step 200 (val_loss=3.94)
- Inference script: `scripts/eval_to_html.py` - generates HTML report with images

## Files
- `scripts/vis_tok_train.py` - Stage 1 training (full model)
- `scripts/vis_mid_train.py` - Stage 2 training (SAM frozen)
- `scripts/eval_to_html.py` - Generate HTML evaluation report
- `scripts/prepare_olmocr.py` - Dataset preparation
