"""
Vision dataloader for nanochat - Karpathy style.

Data format:
  data_dir/train.json  - training samples
  data_dir/val.json    - validation samples (optional, falls back to train.json)
  data_dir/images/     - image files

Sample format:
  {"prompt": "<image>\nOCR this.", "answer": "Hello", "image": "images/doc.png"}
"""

import json
import os
import torch
from PIL import Image
from nanochat.image_process import process_image, count_vision_tokens, expand_image_tokens


def vision_data_loader(B, T, data_dir, device, tokenizer, split="train", base_size=1024):
    """
    Karpathy-style vision data loader.

    Args:
        B: Batch size
        T: Sequence length
        data_dir: Directory containing train.json, val.json, and images/
        device: "cuda" or "cpu"
        tokenizer: tokenizer with encode() method
        split: "train" or "val"
        base_size: Image size for vision encoder

    Yields:
        (inputs, targets, pixel_values) tensors
    """
    # Load samples based on split
    json_file = f"{split}.json"
    json_path = os.path.join(data_dir, json_file)
    if not os.path.exists(json_path):
        # Fallback: use train.json for both (tier-1 overfitting case)
        json_path = os.path.join(data_dir, "train.json")
        if not os.path.exists(json_path):
            # Legacy fallback: overfit_dataset.json
            json_path = os.path.join(data_dir, "overfit_dataset.json")

    with open(json_path, encoding="utf-8") as f:
        samples = json.load(f)

    n_img_tokens = count_vision_tokens(base_size)
    image_token_id = tokenizer.encode_special("<|image|>")

    def process_sample(sample):
        # Replace <image> with <|image|> special token syntax
        prompt_text = sample["prompt"].replace("<image>", "<|image|>")

        # Encode with special token allowed
        prompt_ids = tokenizer.enc.encode(prompt_text, allowed_special={"<|image|>"})

        # Expand the single image token to n_img_tokens copies
        expanded = expand_image_tokens(prompt_ids, image_token_id, n_img_tokens)

        # Encode answer (no special tokens)
        answer_ids = tokenizer.encode(sample["answer"])

        # Load image
        img_path = os.path.join(data_dir, sample["image"])
        pv = process_image(Image.open(img_path), base_size)

        return expanded + answer_ids, pv, len(expanded)

    # Infinite loop over samples
    idx = 0
    while True:
        batch_ids, batch_pv, batch_prompt_lens = [], [], []

        for _ in range(B):
            ids, pv, prompt_len = process_sample(samples[idx])
            batch_ids.append(ids)
            batch_pv.append(pv)
            batch_prompt_lens.append(prompt_len)
            idx = (idx + 1) % len(samples)

        # Collate batch
        max_len = min(max(len(ids) - 1 for ids in batch_ids), T)
        inputs = torch.zeros(B, max_len, dtype=torch.long)
        targets = torch.full((B, max_len), -1, dtype=torch.long)

        for i, (ids, prompt_len) in enumerate(zip(batch_ids, batch_prompt_lens)):
            n = min(len(ids) - 1, max_len)
            t = torch.tensor(ids[:n + 1], dtype=torch.long)
            inputs[i, :n] = t[:-1]
            targets[i, :n] = t[1:]
            targets[i, :prompt_len - 1] = -1  # mask prompt
            targets[i, inputs[i] == image_token_id] = -1  # mask image tokens

        pixel_values = torch.stack(batch_pv)
        yield (inputs.to(device), targets.to(device), pixel_values.to(device))
