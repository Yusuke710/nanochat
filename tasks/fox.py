"""
Fox Benchmark - Page-level OCR evaluation dataset.

From: "Fox: Focus Anywhere for Fine-grained Multi-page Document Understanding"
https://github.com/ucaslcl/Fox

Fox-Page-En: 112 English PDF page images with ground truth OCR text.
Used to evaluate OCR quality of vision-language models.

Usage:
    from tasks.fox import Fox
    dataset = Fox()  # English page OCR (112 samples)
    sample = dataset[0]
    # Returns: {"messages": [...], "images": [PIL.Image]}
"""

from datasets import load_dataset
from tasks.common import Task


class Fox(Task):
    """Fox benchmark for page-level OCR evaluation.

    Loads from HuggingFace: EduardoPacheco/Fox-Page-En
    Contains 112 English PDF page images with ground truth text.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Loading Fox-Page-En dataset...")
        self.ds = load_dataset("EduardoPacheco/Fox-Page-En", split="train")
        # Ground truth loaded separately (conversations format)
        self.gt = load_dataset("ucaslcl/Fox_benchmark_data",
                               data_files="en_page_ocr.json",
                               split="train")
        print(f"  Ready: {len(self.ds)} samples")

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        image = self.ds[index]["image"]
        # Ground truth is in conversations format: [{"from": "human", ...}, {"from": "gpt", "value": "..."}]
        conversations = self.gt[index]["conversations"]
        gt_text = conversations[1]["value"] if len(conversations) > 1 else ""

        return {
            "messages": [
                {"role": "user", "content": "<image>\nOCR this document."},
                {"role": "assistant", "content": gt_text}
            ],
            "images": [image],
        }


if __name__ == "__main__":
    # Quick test
    print("Testing Fox dataset...")
    ds = Fox(stop=3)
    print(f"Loaded {len(ds)} samples")
    ex = ds[0]
    print(f"Keys: {ex.keys()}")
    print(f"User prompt: {ex['messages'][0]['content'][:50]}...")
    print(f"GT length: {len(ex['messages'][1]['content'])} chars")
    print(f"Has image: {ex['images'][0] is not None}")
