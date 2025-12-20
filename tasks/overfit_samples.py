"""
OverfitSamples - Vision task for Tier 1 validation.

Loads samples from data/{split}.json and returns conversation format for
unified multimodal training.

Data format in JSON:
  {"prompt": "<image>\nOCR this.", "answer": "Hello", "image": "images/doc.png"}

Returns conversation format:
  {
      "messages": [
          {"role": "user", "content": "<image>\nOCR this."},
          {"role": "assistant", "content": "Hello"}
      ],
      "image_path": "data/images/doc.png"
  }
"""

import json
import os
from tasks.common import Task


class OverfitSamples(Task):
    """Vision task - loads from data/overfit_samples/{split}.json"""

    def __init__(self, data_dir="data/overfit_samples", split="train", **kwargs):
        super().__init__(**kwargs)
        json_path = os.path.join(data_dir, f"{split}.json")
        if not os.path.exists(json_path):
            # Fallback to train.json for val if not present
            json_path = os.path.join(data_dir, "train.json")
        with open(json_path) as f:
            self.samples = json.load(f)
        self.data_dir = data_dir

    def num_examples(self):
        return len(self.samples)

    def get_example(self, index):
        s = self.samples[index]
        # Convert JSON format to conversation format for render_conversation()
        return {
            "messages": [
                {"role": "user", "content": s["prompt"]},
                {"role": "assistant", "content": s["answer"]}
            ],
            "image_path": os.path.join(self.data_dir, s["image"]),
        }
