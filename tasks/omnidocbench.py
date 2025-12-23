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

Prompts (DeepSeek-OCR style):
    "<image>\nFree OCR."                                    - plain text
    "<image>\n<|grounding|>Convert the document to markdown." - with layout
"""

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from tasks.common import Task


class OmniDocBench(Task):
    """OmniDocBench for document parsing evaluation.

    Loads from HuggingFace: Quivr/OmniDocBench (has annotations in dataset).
    Contains 981 PDF page images with comprehensive annotations.

    Args:
        prompt: User prompt (default: "<image>\nFree OCR.")
        lang: Filter by language ("english", "simplified_chinese", "en_ch_mixed", or None for all)
    """

    def __init__(self, prompt="<image>\nFree OCR.", lang=None, **kwargs):
        super().__init__(**kwargs)
        self.prompt = prompt
        # Load dataset with annotations included
        ds = load_dataset("Quivr/OmniDocBench", "full_dataset", split="train")
        # Filter by language if specified
        if lang:
            ds = ds.filter(lambda x: x["page_info"]["page_attribute"]["language"] == lang)
        self.ds = ds
        print(f"  Ready: {len(self.ds)} samples from OmniDocBench (lang={lang})")

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

        # Extract metadata for category-wise evaluation
        page_attr = page_info.get("page_attribute", {})
        metadata = {
            "language": page_attr.get("language", "unknown"),
            "data_source": page_attr.get("data_source", "unknown"),
            "layout": page_attr.get("layout", "unknown"),
        }

        return {
            "messages": [
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": gt_text}
            ],
            "images": [image] if image else [],
            "metadata": metadata,
        }

    def evaluate(self, conversation, completion):
        """Return 1 - NED (higher is better, per OmniDocBench paper)."""
        gt = conversation["messages"][1]["content"]
        ned = self._normalized_edit_distance(completion, gt)
        return 1 - ned

    @staticmethod
    def _edit_distance(s1, s2):
        """Levenshtein edit distance."""
        if len(s1) < len(s2):
            return OmniDocBench._edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = range(len(s2) + 1)
        for c1 in s1:
            curr = [prev[0] + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
            prev = curr
        return prev[-1]

    @staticmethod
    def _normalized_edit_distance(pred, gt):
        """NED: 0 = perfect, 1 = completely wrong."""
        if len(gt) == 0:
            return 0.0 if len(pred) == 0 else 1.0
        return OmniDocBench._edit_distance(pred, gt) / max(len(pred), len(gt))


if __name__ == "__main__":
    ds = OmniDocBench()
    print(f"OmniDocBench: {len(ds)} samples")
    ex = ds[0]
    print(f"GT length: {len(ex['messages'][1]['content'])} chars")
