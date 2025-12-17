# Vision Sampling Script

## Goal

Create a simple `scripts/vision_sample.py` script to test model quality by running inference on **9 fixed test images** (one from each training dataset), following the `base_loss.py` pattern from nanochat.

This is a **manual sanity check** - you run it to visually inspect model outputs, not part of automated training.

## Test Datasets (9 total)

| # | Dataset | Task Type | Expected Output |
|---|---------|-----------|-----------------|
| 1 | DocBank | Document OCR | Structured text with markdown formatting |
| 2 | olmOCR | PDF → Markdown | Full markdown with headers, lists |
| 3 | LLaVA-CC3M | Image description | Natural language caption |
| 4 | PlotQA | Chart description | Data summary in text |
| 5 | ChartQA | Chart Q&A | Specific answer to question |
| 6 | FigureQA | Yes/No reasoning | "Yes" or "No" |
| 7 | PubTables-1M | Table → HTML | HTML table structure |
| 8 | LaTeX-Formulas | Printed math → LaTeX | LaTeX code |
| 9 | MathWriting | Handwritten math → LaTeX | LaTeX code |

---

## Key Pattern (from base_loss.py)

```python
prompts = ["The capital of France is", ...]
engine = Engine(model, tokenizer)
for prompt in prompts:
    tokens = tokenizer(prompt, prepend="<|bos|>")
    sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
    print0(tokenizer.decode(sample[0]))
```

---

## Implementation Plan

### 1. Create test images directory (one from each dataset)

```
data/test_images/
├── expected.json              # Expected outputs for comparison
├── 01_docbank.png             # DocBank: Document layout OCR
├── 02_olmocr.png              # olmOCR: PDF page to markdown
├── 03_llava_cc3m.jpg          # LLaVA-CC3M: General image description
├── 04_plotqa.png              # PlotQA: Scientific chart description
├── 05_chartqa.png             # ChartQA: Chart question answering
├── 06_figureqa.png            # FigureQA: Yes/No figure reasoning
├── 07_pubtables.jpg           # PubTables-1M: Table to HTML
├── 08_latex_formulas.png      # LaTeX-Formulas: Printed formula to LaTeX
└── 09_mathwriting.png         # MathWriting: Handwritten formula to LaTeX
```

### 2. Create `scripts/vision_sample.py`

```python
# 9 test cases - one from each training dataset
TEST_CASES = [
    # 1. DocBank - Document layout OCR
    ("01_docbank.png", "<image>\nOCR this document page."),

    # 2. olmOCR - PDF to markdown
    ("02_olmocr.png", "<image>\nConvert this document to markdown."),

    # 3. LLaVA-CC3M - General image description
    ("03_llava_cc3m.jpg", "<image>\nDescribe this image in detail."),

    # 4. PlotQA - Chart description
    ("04_plotqa.png", "<image>\nDescribe this chart and its data."),

    # 5. ChartQA - Chart Q&A
    ("05_chartqa.png", "<image>\nWhat is the highest value shown in this chart?"),

    # 6. FigureQA - Yes/No reasoning
    ("06_figureqa.png", "<image>\nIs the blue line always above the red line?"),

    # 7. PubTables-1M - Table to HTML
    ("07_pubtables.jpg", "<image>\nConvert this table to HTML."),

    # 8. LaTeX-Formulas - Printed formula
    ("08_latex_formulas.png", "<image>\nConvert this formula to LaTeX."),

    # 9. MathWriting - Handwritten formula
    ("09_mathwriting.png", "<image>\nConvert this formula to LaTeX."),
]

def main():
    # 1. Load model checkpoint (with fallback for missing vision encoder)
    # 2. Load expected outputs from expected.json
    # 3. For each test case:
    #    - Load image + process through DeepseekOCRProcessor
    #    - Generate output via model.generate()
    #    - Compare with expected output
    #    - Print both for visual comparison
    # 4. Log to report.md
```

### 3. Use model's `generate()` directly

No need for separate VisionEngine - just use `nano_deepseek_ocr.py`'s built-in `generate()` method:

```python
output = model.generate(
    input_ids=input_ids,
    pixel_values=pixel_values,
    images_crop=images_crop,
    images_spatial_crop=images_spatial_crop,
    max_tokens=256,
    temperature=0.0,
)
```

### 4. Modify `image_process.py`

Add `prompt` parameter to `tokenize_with_images()`:

```python
def tokenize_with_images(self, images, prompt=None, ...):
    conversation = prompt if prompt else PROMPT  # Currently hardcoded
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `scripts/vision_sample.py` | CREATE - Main script |
| `process/image_process.py` | MODIFY - Add prompt parameter |
| `data/test_images/` | CREATE - Directory with 9 test images (one per dataset) |
| `data/test_images/expected.json` | CREATE - Expected outputs for comparison |

---

## Usage

```bash
# Basic
python -m scripts.vision_sample

# With specific checkpoint step
python -m scripts.vision_sample --model_step 10000

# Limit tokens for faster testing
python -m scripts.vision_sample --max_tokens 64
```

---

## Output Format (with Expected vs Generated)

```
============================================================
Vision Sampling Results (9 datasets)
============================================================

[1/9] DocBank - Document Layout OCR
[IMAGE] 01_docbank.png
[PROMPT] <image>\nOCR this document page.
[EXPECTED]
## Abstract

We present a novel approach to document understanding...

[GENERATED]
## Abstract

We present a novel approach to document understanding...

[MATCH] EXACT
----------------------------------------
[2/9] olmOCR - PDF to Markdown
[IMAGE] 02_olmocr.png
[PROMPT] <image>\nConvert this document to markdown.
...
----------------------------------------
[8/9] LaTeX-Formulas - Printed Formula
[IMAGE] 08_latex_formulas.png
[PROMPT] <image>\nConvert this formula to LaTeX.
[EXPECTED]
\frac{1}{2} \sum_{i=1}^{n} x_i^2

[GENERATED]
\frac{1}{2} \sum_{i=1}^{n} x_i^2

[MATCH] EXACT
----------------------------------------
[9/9] MathWriting - Handwritten Formula
[IMAGE] 09_mathwriting.png
[PROMPT] <image>\nConvert this formula to LaTeX.
[EXPECTED]
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}

[GENERATED]
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}

[MATCH] EXACT
============================================================
Summary: 8/9 EXACT, 1/9 PARTIAL
============================================================
```

---

## Expected Outputs File (`data/test_images/expected.json`)

```json
{
  "01_docbank.png": {
    "dataset": "DocBank",
    "prompt": "<image>\nOCR this document page.",
    "expected": "## Abstract\n\nWe present a novel approach..."
  },
  "02_olmocr.png": {
    "dataset": "olmOCR",
    "prompt": "<image>\nConvert this document to markdown.",
    "expected": "# Introduction\n\nThis paper describes..."
  },
  "03_llava_cc3m.jpg": {
    "dataset": "LLaVA-CC3M",
    "prompt": "<image>\nDescribe this image in detail.",
    "expected": "A golden retriever dog playing in a park..."
  },
  "04_plotqa.png": {
    "dataset": "PlotQA",
    "prompt": "<image>\nDescribe this chart and its data.",
    "expected": "This line chart shows GDP growth from 2010-2020..."
  },
  "05_chartqa.png": {
    "dataset": "ChartQA",
    "prompt": "<image>\nWhat is the highest value shown in this chart?",
    "expected": "The highest value is 85.3% in 2019."
  },
  "06_figureqa.png": {
    "dataset": "FigureQA",
    "prompt": "<image>\nIs the blue line always above the red line?",
    "expected": "No"
  },
  "07_pubtables.jpg": {
    "dataset": "PubTables-1M",
    "prompt": "<image>\nConvert this table to HTML.",
    "expected": "<table><tr><th>Year</th><th>Value</th></tr><tr><td>2020</td><td>100</td></tr></table>"
  },
  "08_latex_formulas.png": {
    "dataset": "LaTeX-Formulas",
    "prompt": "<image>\nConvert this formula to LaTeX.",
    "expected": "\\frac{1}{2} \\sum_{i=1}^{n} x_i^2"
  },
  "09_mathwriting.png": {
    "dataset": "MathWriting",
    "prompt": "<image>\nConvert this formula to LaTeX.",
    "expected": "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}"
  }
}
```

---

## Graceful Fallbacks

- Missing test images → Skip with warning
- Missing vision encoder → Use zeros (text-only mode)
- Missing checkpoint → Create dummy model for pipeline testing

---

## Dependencies

- Reuses: `nanochat.checkpoint_manager`, `nanochat.common`, `nanochat.report`
- Requires: `nano_deepseek_ocr` model class (from blueprint)
- Requires: `DeepseekOCRProcessor` (already in reference_code)
