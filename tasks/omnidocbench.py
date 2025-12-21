"""
OmniDocBench - Comprehensive document parsing benchmark.

From: "OmniDocBench: Benchmarking Diverse PDF Document Parsing" (CVPR 2025)
https://github.com/opendatalab/OmniDocBench

981 PDF pages covering 9 document types, 4 layout types, 3 languages.
Rich annotations: text blocks, tables, formulas, reading order.

Usage:
    from tasks.omnidocbench import OmniDocBench
    dataset = OmniDocBench()
    sample = dataset[0]
    # Returns: {"messages": [...], "images": [PIL.Image], "metadata": {...}}
"""

from io import BytesIO
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from tasks.common import Task


class OmniDocBench(Task):
    """OmniDocBench for document parsing evaluation.

    Loads from HuggingFace: Quivr/OmniDocBench (has annotations in dataset).
    Contains 981 PDF page images with comprehensive annotations.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load dataset with annotations included
        self.ds = load_dataset("Quivr/OmniDocBench", "full_dataset", split="train")
        print(f"  Ready: {len(self.ds)} samples from OmniDocBench")

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        layout_dets = row.get("layout_dets", [])
        page_info = row.get("page_info", {})
        image_path = page_info.get("image_path", "")

        # Download image from HuggingFace
        image = None
        if image_path:
            try:
                img_file = hf_hub_download(
                    repo_id="Quivr/OmniDocBench",
                    filename=f"images/{image_path}",
                    repo_type="dataset"
                )
                image = Image.open(img_file).convert("RGB")
            except Exception:
                pass  # Image not found

        # Build ground truth text from text blocks in reading order
        text_blocks = []
        for det in layout_dets:
            if det.get("category_type") in ["text_block", "title", "header", "footer"]:
                text = det.get("text", "")
                if text:
                    order = det.get("order")
                    order = order if order is not None else 999
                    text_blocks.append((order, text))

        # Sort by reading order and join
        text_blocks.sort(key=lambda x: x[0])
        gt_text = "\n\n".join([t[1] for t in text_blocks])

        return {
            "messages": [
                {"role": "user", "content": "<image>\nOCR this document."},
                {"role": "assistant", "content": gt_text}
            ],
            "images": [image] if image else [],
        }

    def evaluate(self, conversation, completion):
        """Return 1 - NED (higher is better, per OmniDocBench paper)."""
        from nanochat.vision_eval import normalized_edit_distance
        gt = conversation["messages"][1]["content"]
        return 1 - normalized_edit_distance(completion, gt)


class OmniDocBenchFull(Task):
    """OmniDocBench with full annotations (tables, formulas, etc).

    Returns structured output including:
    - text_blocks: paragraphs and titles
    - tables: LaTeX/HTML table representations
    - formulas: LaTeX formula representations
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Loading OmniDocBench dataset (full mode)...")
        self.ds = load_dataset("opendatalab/OmniDocBench", split="train")
        print(f"  Ready: {len(self.ds)} samples")

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        image = row.get("image")
        layout_dets = row.get("layout_dets", [])
        page_info = row.get("page_info", {})

        # Categorize elements
        text_blocks, tables, formulas = [], [], []
        for det in layout_dets:
            cat = det.get("category_type", "")
            order = det.get("order", 999)

            if cat in ["text_block", "title"]:
                text_blocks.append((order, det.get("text", "")))
            elif cat == "table":
                tables.append((order, det.get("latex", det.get("html", ""))))
            elif cat in ["equation_isolated", "equation_inline"]:
                formulas.append((order, det.get("latex", "")))

        # Sort each by reading order
        text_blocks.sort(key=lambda x: x[0])
        tables.sort(key=lambda x: x[0])
        formulas.sort(key=lambda x: x[0])

        page_attr = page_info.get("page_attribute", {})

        return {
            "messages": [
                {"role": "user", "content": "<image>\nParse this document."},
                {"role": "assistant", "content": ""}  # filled during structured eval
            ],
            "images": [image] if image else [],
            "annotations": {
                "text_blocks": [t[1] for t in text_blocks],
                "tables": [t[1] for t in tables],
                "formulas": [t[1] for t in formulas],
            },
            "metadata": {
                "page_no": page_info.get("page_no", 0),
                "data_source": page_attr.get("data_source", "unknown"),
                "language": page_attr.get("language", "unknown"),
                "layout": page_attr.get("layout", "unknown"),
            }
        }


if __name__ == "__main__":
    ds = OmniDocBench()
    print(f"OmniDocBench: {len(ds)} samples")
    ex = ds[0]
    print(f"GT length: {len(ex['messages'][1]['content'])} chars")
