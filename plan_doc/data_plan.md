# Data Plan

## Data Format

```python
# Vision sample
{"text": "<image>\nOCR this.", "response": "Hello world", "image_path": "doc.png"}

# Text sample
{"text": "What is 2+2?", "response": "4"}
```

## Datasets (8 total)

| Dataset | HuggingFace ID | Task |
|---------|----------------|------|
| DocBank | `liminghao1630/DocBank` | Document OCR |
| olmOCR | `allenai/olmOCR-mix-1025` | PDF → Markdown |
| LLaVA-CC3M | `liuhaotian/LLaVA-CC3M-Pretrain-595K` | Image description |
| PlotQA | `achang/plot_qa` | Chart description |
| ChartQA | `HuggingFaceM4/ChartQA` | Chart Q&A |
| FigureQA | `vikhyatk/figureqa` | Yes/No reasoning |
| PubTables-1M | `bsmock/pubtables-1m` | Table → HTML |
| LaTeX-Formulas | `OleehyO/latex-formulas` | Printed math → LaTeX |

## Task Class Template

```python
class ExampleTask(Task):
    def __init__(self, split="train"):
        self.ds = load_dataset("huggingface/id", split=split)

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        return {
            "text": f"<image>\n{row['prompt']}",
            "response": row["answer"],
            "image_path": f"data/example/{index:08d}.png"
        }
```

## Training Mixture

```python
# Stage 1: Vision only
stage1_ds = TaskMixture([DocBank(), OlmOCR(), LLaVACC3M(), PlotQA(), ...])

# Stage 2: Vision + Text
stage2_ds = TaskMixture([
    DocBank(), OlmOCR(), FigureQA,
    LLaVACC3M(), PlotQA(), ChartQA(), LaTeXFormulas()
    SmolTalk(stop=50000),  # 10% text
])
```

## Loss Strategy

**Pre-training loss**: Supervise ALL text tokens, mask only image positions.

```python
targets = ids[1:]  # next-token prediction
for start, end in image_positions:
    targets[start:end-1] = -1  # ignore image tokens
```