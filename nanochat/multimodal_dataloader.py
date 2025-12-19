"""
Unified multimodal dataloader with PyTorch DataLoader for high MFU.

One tokenizer, one loader for all modalities. Change TaskMixture contents, not the code.

Usage:
    # Stage 1: vision only
    train_ds = TaskMixture([OverfitSamples(data_dir="data")])

    # Stage 2: vision + text
    train_ds = TaskMixture([OverfitSamples(...), SmolTalk(...)])

    # Same loader for both - returns DataLoader iterator
    train_loader = create_multimodal_loader(train_ds, tokenizer, B, T, base_size)
    for inputs, targets, media in train_loader:
        loss = model(input_ids=inputs, targets=targets, pixel_values=media["pixel_values"])
"""

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from nanochat.image_process import process_image, count_vision_tokens, expand_image_tokens


class MultimodalDataset(Dataset):
    """PyTorch Dataset wrapper for TaskMixture with unified tokenization."""

    def __init__(self, task_mixture, tokenizer, base_size=1024):
        self.task_mixture = task_mixture
        self.tokenizer = tokenizer
        self.base_size = base_size
        self.n_img_tokens = count_vision_tokens(base_size)
        self.image_token_id = tokenizer.encode_special("<|image|>")

    def __len__(self):
        return len(self.task_mixture)

    def __getitem__(self, idx):
        sample = self.task_mixture[idx]

        # Tokenize using render_conversation (handles media placeholders)
        ids, _ = self.tokenizer.render_conversation(sample, max_tokens=16384)

        # Handle image if present
        pixel_values = None
        if "image_path" in sample and sample["image_path"]:
            ids = expand_image_tokens(ids, self.image_token_id, self.n_img_tokens)
            pixel_values = process_image(Image.open(sample["image_path"]), self.base_size)

        return ids, pixel_values, self.image_token_id


def multimodal_collate_fn(batch, T, base_size):
    """Collate batch into padded tensors with media dict."""
    B = len(batch)

    # Find max sequence length in batch (capped at T)
    max_len = min(max(len(ids) - 1 for ids, _, _ in batch), T)

    inputs = torch.zeros(B, max_len, dtype=torch.long)
    targets = torch.full((B, max_len), -1, dtype=torch.long)

    pixels_list = []
    batch_has_image = False

    for i, (ids, pixel_values, image_token_id) in enumerate(batch):
        n = min(len(ids) - 1, max_len)
        t = torch.tensor(ids[:n + 1], dtype=torch.long)
        inputs[i, :n] = t[:-1]
        targets[i, :n] = t[1:]
        # Mask image token positions
        targets[i, inputs[i] == image_token_id] = -1

        if pixel_values is not None:
            pixels_list.append(pixel_values)
            batch_has_image = True

    # Build media dict
    if batch_has_image:
        # Pad with zeros for text-only samples in mixed batch
        while len(pixels_list) < B:
            pixels_list.append(torch.zeros(3, base_size, base_size))
        media = {"pixel_values": torch.stack(pixels_list)}
    else:
        media = {"pixel_values": None}

    return inputs, targets, media


def create_multimodal_loader(task_mixture, tokenizer, B, T, base_size, num_workers=4):
    """Create multimodal DataLoader with worker-side processing.

    Args:
        task_mixture: TaskMixture containing vision and/or text tasks
        tokenizer: RustBPETokenizer with render_conversation()
        B: batch size
        T: sequence length
        base_size: image resolution (e.g., 1024)
        num_workers: number of worker processes for data loading

    Returns:
        DataLoader that yields (inputs, targets, media) tuples
        - inputs: (B, T) tensor of input token ids
        - targets: (B, T) tensor of target token ids (-1 for masked positions)
        - media: dict with "pixel_values" (B, 3, H, W) or None for text-only batches

    Automatically uses DistributedSampler when running in DDP mode.
    """
    dataset = MultimodalDataset(task_mixture, tokenizer, base_size)

    # Use DistributedSampler if in DDP mode
    ddp = dist.is_initialized()
    sampler = DistributedSampler(dataset, shuffle=True) if ddp else None
    shuffle = False if ddp else True

    return DataLoader(
        dataset,
        batch_size=B,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        collate_fn=lambda batch: multimodal_collate_fn(batch, T, base_size),
        drop_last=True,
    )
