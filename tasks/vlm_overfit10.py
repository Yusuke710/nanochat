"""
VLM Overfit 10 - Vision samples for VLM validation.

10 samples from SROIE, ChartQA, TextVQA for quick validation
of vision encoder training before scaling.

Data format on HuggingFace (ImageFolder with metadata.jsonl):
  {"file_name": "images/receipt_000.jpg", "id": "...", "source": "SROIE",
   "type": "receipt_ocr", "prompt": "...", "answer": "..."}

Returns conversation format with image_path:
  {
      "messages": [
          {"role": "user", "content": "<image>\nOCR this."},
          {"role": "assistant", "content": "..."}
      ],
      "image_path": "/path/to/cached/image.jpg"
  }
"""

from datasets import load_dataset
from tasks.common import Task


class VLMOverfit10(Task):
    """VLM overfit dataset. 10 samples from SROIE, ChartQA, TextVQA.

    Uses ImageFolder format on HuggingFace - images stored as files.
    Downloads once to cache, then loads by path for efficiency.
    """

    def __init__(self, split="train", repo_id="USERNAME/vlm-overfit10", **kwargs):
        super().__init__(**kwargs)
        # Download dataset to cache (images as files, not embedded in Parquet)
        # Single split only - overfit dataset uses train for everything
        self.ds = load_dataset(repo_id, split="train")

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        # HuggingFace ImageFolder returns PIL Image with .filename attribute
        # pointing to the cached file path
        image = row["image"]
        image_path = getattr(image, "filename", None)

        return {
            "messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["answer"]}
            ],
            # Path-based loading for scalability
            "image_path": image_path,
        }
