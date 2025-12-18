"""
Vision dataloader for nanochat.

Data format:
  data_dir/train.json  - training samples
  data_dir/val.json    - validation samples
  data_dir/images/     - image files

Sample format:
  {"prompt": "<image>\nOCR this.", "answer": "Hello", "image": "images/doc.png"}
"""

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from nanochat.image_process import process_image, count_vision_tokens, expand_image_tokens


class VisionDataset(Dataset):
    def __init__(self, data_dir, tokenizer, split="train", base_size=1024):
        json_path = os.path.join(data_dir, f"{split}.json")
        if not os.path.exists(json_path):
            json_path = os.path.join(data_dir, "train.json")
        with open(json_path, encoding="utf-8") as f:
            self.samples = json.load(f)

        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.base_size = base_size
        self.n_img_tokens = count_vision_tokens(base_size)
        self.image_token_id = tokenizer.encode_special("<|image|>")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Tokenize prompt with image token expansion
        prompt_text = sample["prompt"].replace("<image>", "<|image|>")
        prompt_ids = self.tokenizer.enc.encode(prompt_text, allowed_special={"<|image|>"})
        expanded = expand_image_tokens(prompt_ids, self.image_token_id, self.n_img_tokens)
        answer_ids = self.tokenizer.encode(sample["answer"])

        # Load image (done in worker process)
        img_path = os.path.join(self.data_dir, sample["image"])
        pixel_values = process_image(Image.open(img_path), self.base_size)

        return expanded + answer_ids, len(expanded), pixel_values, self.image_token_id


def collate_fn(batch, T=4096):
    """Collate batch into padded tensors."""
    B = len(batch)
    max_len = min(max(len(ids) - 1 for ids, _, _, _ in batch), T)

    inputs = torch.zeros(B, max_len, dtype=torch.long)
    targets = torch.full((B, max_len), -1, dtype=torch.long)
    pixel_values = torch.stack([pv for _, _, pv, _ in batch])

    for i, (ids, prompt_len, _, image_token_id) in enumerate(batch):
        n = min(len(ids) - 1, max_len)
        t = torch.tensor(ids[:n + 1], dtype=torch.long)
        inputs[i, :n] = t[:-1]
        targets[i, :n] = t[1:]
        targets[i, :prompt_len - 1] = -1  # mask prompt
        targets[i, inputs[i] == image_token_id] = -1  # mask image tokens

    return inputs, targets, pixel_values


def create_vision_loader(B, T, data_dir, tokenizer, split="train", base_size=1024, num_workers=4):
    """Create vision DataLoader with worker-side processing."""
    dataset = VisionDataset(data_dir, tokenizer, split, base_size)
    return DataLoader(
        dataset,
        batch_size=B,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        collate_fn=lambda batch: collate_fn(batch, T),
        drop_last=True,
    )
