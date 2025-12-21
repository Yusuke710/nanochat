"""
Fox Benchmark - Page-level OCR evaluation dataset.

From: "Fox: Focus Anywhere for Fine-grained Multi-page Document Understanding"
https://github.com/ucaslcl/Fox

English page OCR: 112 samples (same as DeepSeek-OCR evaluation).
"""

import json
import random
import zipfile
from io import BytesIO
from PIL import Image
from huggingface_hub import hf_hub_download
from tasks.common import Task


class Fox(Task):
    """Fox benchmark for page-level OCR evaluation (112 samples)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Download and load from zip
        zip_path = hf_hub_download(
            repo_id="ucaslcl/Fox_benchmark_data",
            filename="focus_benchmark_test.zip",
            repo_type="dataset"
        )
        self.zip_path = zip_path

        # Load ground truth JSON
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open("focus_benchmark_test/en_page_ocr.json") as f:
                self.data = json.load(f)

        # Shuffle with fixed seed (same as MMLU/ARC)
        rng = random.Random(42)
        rng.shuffle(self.data)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.data)

    def get_example(self, index):
        item = self.data[index]
        image_name = item["image"]
        conversations = item["conversations"]
        gt_text = conversations[1]["value"] if len(conversations) > 1 else ""

        # Load image from zip
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            img_path = f"focus_benchmark_test/en_pdf_png/{image_name}"
            with z.open(img_path) as f:
                image = Image.open(BytesIO(f.read())).convert("RGB")

        return {
            "messages": [
                {"role": "user", "content": "<image>\nOCR this document."},
                {"role": "assistant", "content": gt_text}
            ],
            "images": [image],
        }

    def evaluate(self, conversation, completion):
        """Return precision score (DeepSeek-OCR metric)."""
        from nanochat.vision_eval import precision
        gt = conversation["messages"][1]["content"]
        return precision(completion, gt)


if __name__ == "__main__":
    ds = Fox()
    print(f"Fox: {len(ds)} samples")
