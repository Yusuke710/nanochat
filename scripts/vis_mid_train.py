"""
Vision Mid Training (Stage 2) - Train with frozen SAM and projector.

Stage 2 freezes:
- SAM encoder (sam_model)
- Projector (linear layer that maps 2048 -> n_embd)

Stage 2 trains:
- CLIP encoder (vision_model)
- GPT (language model)
- Special tokens (image_newline, view_separator)

Usage:
    python -m scripts.vis_mid_train --tier1          # Tier 1: Overfit on 10 images
    python -m scripts.vis_mid_train --checkpoint checkpoints/tier1_step_300.pt
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Optional
from functools import partial

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

from nanochat.gpt import GPTConfig
from nanochat.nano_deepseek_ocr import NanoDeepseekOCR, build_nano_deepseek_ocr
from nanochat.deepencoder.load_pretrained import (
    load_sam_weights_from_hf,
    load_clip_weights_from_hf,
    load_nanochat_gpt_from_hf,
)
from nanochat.image_process import process_image, count_vision_tokens, expand_image_tokens
from nanochat.tokenizer import RustBPETokenizer
from PIL import Image


def load_overfit_dataset(data_dir: str = "data"):
    """Load the overfit dataset from JSON."""
    json_path = Path(data_dir) / "overfit_dataset.json"
    with open(json_path, "r") as f:
        return json.load(f)


class OverfitDataset(Dataset):
    """Dataset for overfitting on a few images with caching."""

    def __init__(self, data_dir: str = "data", tokenizer=None, image_token_id: int = None, cache_images: bool = True):
        self.data_dir = Path(data_dir)
        self.samples = load_overfit_dataset(data_dir)
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id
        self.cache_images = cache_images

        # Compute vision token count
        self.num_vision_tokens = count_vision_tokens(base_size=1024)

        # Cache for processed images (for tier-1 with small dataset)
        self._image_cache = {}
        self._token_cache = {}

        if cache_images:
            print(f"Caching {len(self.samples)} images in memory...")
            self._preload_all()

    def _preload_all(self):
        """Preload all images and tokenize all samples."""
        for idx, sample in enumerate(self.samples):
            # Load and process image
            image_path = self.data_dir / sample["image"]
            image = Image.open(image_path).convert("RGB")
            pixel_values = process_image(image, base_size=1024)
            self._image_cache[idx] = pixel_values.squeeze(0)

            # Tokenize
            prompt_text = sample["prompt"].replace("<image>", "").strip()
            prompt_tokens = self.tokenizer.encode(prompt_text)
            answer_tokens = self.tokenizer.encode(sample["answer"])

            # Build full sequence
            tokens = [self.image_token_id] * self.num_vision_tokens + prompt_tokens + answer_tokens
            prompt_len = self.num_vision_tokens + len(prompt_tokens)

            input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
            targets = torch.tensor(tokens[1:], dtype=torch.long)
            targets[:prompt_len - 1] = -1

            self._token_cache[idx] = {
                "input_ids": input_ids,
                "targets": targets,
            }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.cache_images and idx in self._image_cache:
            return {
                "input_ids": self._token_cache[idx]["input_ids"],
                "targets": self._token_cache[idx]["targets"],
                "pixel_values": self._image_cache[idx],
                "id": self.samples[idx]["id"],
            }

        # Non-cached path
        sample = self.samples[idx]
        image_path = self.data_dir / sample["image"]
        image = Image.open(image_path).convert("RGB")
        pixel_values = process_image(image, base_size=1024)

        prompt_text = sample["prompt"].replace("<image>", "").strip()
        prompt_tokens = self.tokenizer.encode(prompt_text)
        answer_tokens = self.tokenizer.encode(sample["answer"])

        tokens = [self.image_token_id] * self.num_vision_tokens + prompt_tokens + answer_tokens
        prompt_len = self.num_vision_tokens + len(prompt_tokens)

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        targets = torch.tensor(tokens[1:], dtype=torch.long)
        targets[:prompt_len - 1] = -1

        return {
            "input_ids": input_ids,
            "targets": targets,
            "pixel_values": pixel_values.squeeze(0),
            "id": sample["id"],
        }


def collate_fn(batch, pad_token_id=0, max_seq_len=4096):
    """Collate batch with padding."""
    batch_size = len(batch)

    # Find max length in batch
    max_len = min(max(len(b["input_ids"]) for b in batch), max_seq_len)

    # Initialize tensors
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    targets = torch.full((batch_size, max_len), -1, dtype=torch.long)
    pixel_values = torch.stack([b["pixel_values"] for b in batch])

    for i, b in enumerate(batch):
        seq_len = min(len(b["input_ids"]), max_len)
        input_ids[i, :seq_len] = b["input_ids"][:seq_len]
        targets[i, :seq_len] = b["targets"][:seq_len]

    return {
        "input_ids": input_ids,
        "targets": targets,
        "pixel_values": pixel_values,
    }


def freeze_sam_and_projector(model):
    """
    Freeze SAM encoder and projector for Stage 2 training.

    Frozen:
    - model.sam_model (SAM ViT-B encoder + compression layers)
    - model.projector (linear layer 2048 -> n_embd)

    Trainable:
    - model.vision_model (CLIP ViT-L/14)
    - model.gpt (GPT language model)
    - model.image_newline (special token embedding)
    - model.view_separator (special token embedding)
    """
    # Freeze SAM
    for param in model.sam_model.parameters():
        param.requires_grad = False

    # Freeze projector
    for param in model.projector.parameters():
        param.requires_grad = False

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n=== Stage 2 Freezing ===")
    print(f"Frozen: SAM encoder + projector")
    print(f"Trainable: CLIP encoder + GPT + special tokens")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Frozen params: {frozen_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params*100:.1f}%")

    return model


def train_tier1(args):
    """Train on tier-1 overfit dataset with Stage 2 freezing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load HF token
    hf_token = os.getenv("HF_TOKEN")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = RustBPETokenizer.from_directory("tokenizer")
    vocab_size = tokenizer.get_vocab_size()
    print(f"Loaded tokenizer with vocab size: {vocab_size}")

    image_token_id = vocab_size - 1
    print(f"Using image token ID: {image_token_id}")

    # Build model
    print("Building model...")
    gpt_config = GPTConfig(
        sequence_len=4096,
        vocab_size=vocab_size,
        n_layer=20,
        n_head=16,
        n_kv_head=16,
        n_embd=1280,
    )

    model = build_nano_deepseek_ocr(gpt_config=gpt_config)
    model.set_image_token_id(image_token_id)

    # Load weights - either from checkpoint or from pretrained
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
        print("Checkpoint loaded (includes trained projector from Stage 1)")
    else:
        print("Loading pretrained weights (fresh start)...")
        model.sam_model = load_sam_weights_from_hf(model.sam_model, hf_token=hf_token, verbose=False)
        model.vision_model = load_clip_weights_from_hf(model.vision_model, hf_token=hf_token, verbose=False)
        model.gpt = load_nanochat_gpt_from_hf(model.gpt, hf_token=hf_token, verbose=False)

    model = model.to(device)

    # Apply Stage 2 freezing (freeze SAM and projector)
    model = freeze_sam_and_projector(model)

    # Create dataset with caching
    print("\nLoading dataset...")
    dataset = OverfitDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        image_token_id=image_token_id,
        cache_images=True,  # Cache all images for tier-1
    )
    print(f"Dataset has {len(dataset)} samples")
    print(f"Vision tokens per image: {dataset.num_vision_tokens}")

    # Create DataLoader
    collate = partial(collate_fn, pad_token_id=0, max_seq_len=4096)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,  # Use 0 for cached dataset
        pin_memory=True,
    )

    # Setup optimizer - only for trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.steps
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)

    # Training loop
    print(f"\nStarting Stage 2 training for {total_steps} steps (batch_size={args.batch_size})...")
    model.train()

    step = 0
    running_loss = 0.0
    log_interval = 10

    start_time = time.time()

    while step < total_steps:
        for batch in dataloader:
            if step >= total_steps:
                break

            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(
                    input_ids=input_ids,
                    targets=targets,
                    pixel_values=pixel_values,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            step += 1

            if step % log_interval == 0:
                avg_loss = running_loss / log_interval
                elapsed = time.time() - start_time
                lr = scheduler.get_last_lr()[0]
                samples_per_sec = (step * args.batch_size) / elapsed
                print(f"Step {step}/{total_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                      f"Time: {elapsed:.1f}s | {samples_per_sec:.1f} samples/s")
                running_loss = 0.0

                if avg_loss < 0.01:
                    print(f"\nReached near-zero loss ({avg_loss:.6f})!")

    # Save checkpoint
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"stage2_step_{step}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nCheckpoint saved to {checkpoint_path}")

    # Final evaluation (batched)
    print("\n=== Final Evaluation ===")
    model.eval()

    eval_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        pin_memory=True,
    )

    total_loss = 0.0
    sample_idx = 0
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Get per-sample loss
                loss = model(input_ids=input_ids, targets=targets, pixel_values=pixel_values, loss_reduction='none')
                # Average loss per sample (ignoring -1 targets)
                mask = (targets != -1).float()
                per_sample_loss = (loss.view(targets.shape) * mask).sum(dim=1) / mask.sum(dim=1)

            for i, l in enumerate(per_sample_loss):
                if sample_idx < len(dataset.samples):
                    print(f"  {dataset.samples[sample_idx]['id']}: loss = {l.item():.4f}")
                    total_loss += l.item()
                    sample_idx += 1

    avg_loss = total_loss / len(dataset)
    print(f"\nAverage loss on training set: {avg_loss:.4f}")

    if avg_loss < 0.1:
        print("SUCCESS: Stage 2 tier-1 achieved!")
    else:
        print("INCOMPLETE: Loss is still too high.")


def main():
    parser = argparse.ArgumentParser(description="Vision Mid Training (Stage 2) - Frozen SAM + Projector")
    parser.add_argument("--tier1", action="store_true", help="Run tier-1 overfitting")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to Stage 1 checkpoint (to reuse projector)")
    parser.add_argument("--steps", type=int, default=300, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")

    args = parser.parse_args()

    if args.tier1:
        train_tier1(args)
    else:
        print("Please specify --tier1 for tier-1 overfitting")
        print("Full training not yet implemented")


if __name__ == "__main__":
    main()
