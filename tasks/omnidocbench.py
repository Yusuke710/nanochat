"""
OmniDocBench - Comprehensive document parsing benchmark.

From: "OmniDocBench: Benchmarking Diverse PDF Document Parsing" (CVPR 2025)
https://github.com/opendatalab/OmniDocBench

1355 PDF pages covering 9 document types, 4 layout types, 3 languages.
Rich annotations: text blocks, tables, formulas, reading order.

Usage:
    from tasks.omnidocbench import OmniDocBench
    dataset = OmniDocBench()
    sample = dataset[0]
    # Returns: {"messages": [...], "images": [PIL.Image], "metadata": {...}}
"""

import json
from datasets import load_dataset
from tasks.common import Task


class OmniDocBench(Task):
    """OmniDocBench for document parsing evaluation.

    Loads from HuggingFace: opendatalab/OmniDocBench
    Contains 1355 PDF page images with comprehensive annotations.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Loading OmniDocBench dataset...")
        self.ds = load_dataset("opendatalab/OmniDocBench", split="train")
        print(f"  Ready: {len(self.ds)} samples")

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]

        # Extract image
        image = row.get("image")

        # Parse annotations - layout_dets contains all page elements
        layout_dets = row.get("layout_dets", [])
        page_info = row.get("page_info", {})

        # Build ground truth text from text blocks in reading order
        text_blocks = []
        for det in layout_dets:
            if det.get("category_type") in ["text_block", "title"]:
                text = det.get("text", "")
                if text:
                    text_blocks.append((det.get("order", 999), text))

        # Sort by reading order and join
        text_blocks.sort(key=lambda x: x[0])
        gt_text = "\n\n".join([t[1] for t in text_blocks])

        # Page attributes for filtering/analysis
        page_attr = page_info.get("page_attribute", {})

        return {
            "messages": [
                {"role": "user", "content": "<image>\nOCR this document."},
                {"role": "assistant", "content": gt_text}
            ],
            "images": [image] if image else [],
            "metadata": {
                "page_no": page_info.get("page_no", 0),
                "data_source": page_attr.get("data_source", "unknown"),
                "language": page_attr.get("language", "unknown"),
                "layout": page_attr.get("layout", "unknown"),
            }
        }


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
    # Quick test
    print("Testing OmniDocBench dataset...")
    ds = OmniDocBench(stop=3)
    print(f"Loaded {len(ds)} samples")
    ex = ds[0]
    print(f"Keys: {ex.keys()}")
    print(f"GT length: {len(ex['messages'][1]['content'])} chars")
    print(f"Metadata: {ex['metadata']}")
    print(f"Has image: {len(ex['images']) > 0}")
