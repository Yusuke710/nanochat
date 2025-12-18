"""
Vision Sample - Sanity check script for nano-deepseek-ocr

Usage:
    python -m scripts.vision_sample                    # Run with latest checkpoint
    python -m scripts.vision_sample --step 1000        # Run with specific step
    python -m scripts.vision_sample --mock             # Test script without model
    python -m scripts.vision_sample --batch-loss       # Fast batched loss eval only

Loads 10 test images from data/overfit_dataset.json and shows EXPECTED vs GENERATED.
"""

import argparse
import json
import time
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def load_dataset(data_dir: str = "data") -> list[dict]:
    """Load overfit dataset from JSON."""
    json_path = Path(data_dir) / "overfit_dataset.json"
    with open(json_path, "r") as f:
        return json.load(f)


class EvalDataset(Dataset):
    """Dataset for batched evaluation."""

    def __init__(self, data_dir: str, samples: list, tokenizer, image_token_id: int):
        from nanochat.image_process import process_image, count_vision_tokens
        from PIL import Image

        self.data_dir = Path(data_dir)
        self.samples = samples
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id
        self.num_vision_tokens = count_vision_tokens(base_size=1024)

        # Pre-cache everything
        self._cache = []
        print(f"Caching {len(samples)} samples...")
        for sample in samples:
            image_path = self.data_dir / sample["image"]
            image = Image.open(image_path).convert("RGB")
            pixel_values = process_image(image, base_size=1024).squeeze(0)

            prompt_text = sample["prompt"].replace("<image>", "").strip()
            prompt_tokens = tokenizer.encode(prompt_text)
            answer_tokens = tokenizer.encode(sample["answer"])

            tokens = [image_token_id] * self.num_vision_tokens + prompt_tokens + answer_tokens
            prompt_len = self.num_vision_tokens + len(prompt_tokens)

            input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
            targets = torch.tensor(tokens[1:], dtype=torch.long)
            targets[:prompt_len - 1] = -1

            self._cache.append({
                "input_ids": input_ids,
                "targets": targets,
                "pixel_values": pixel_values,
                "prompt_len": prompt_len,
                "id": sample["id"],
            })

    def __len__(self):
        return len(self._cache)

    def __getitem__(self, idx):
        return self._cache[idx]


def collate_fn(batch, pad_token_id=0):
    """Collate batch with padding."""
    batch_size = len(batch)
    max_len = max(len(b["input_ids"]) for b in batch)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    targets = torch.full((batch_size, max_len), -1, dtype=torch.long)
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    prompt_lens = [b["prompt_len"] for b in batch]

    for i, b in enumerate(batch):
        seq_len = len(b["input_ids"])
        input_ids[i, :seq_len] = b["input_ids"]
        targets[i, :seq_len] = b["targets"]

    return {
        "input_ids": input_ids,
        "targets": targets,
        "pixel_values": pixel_values,
        "prompt_lens": prompt_lens,
    }


def load_model(checkpoint_path: str = None, step: int = None):
    """Load nano-deepseek-ocr model."""
    from nanochat.gpt import GPTConfig
    from nanochat.nano_deepseek_ocr import build_nano_deepseek_ocr
    from nanochat.tokenizer import RustBPETokenizer

    # Determine checkpoint path
    if checkpoint_path is None:
        checkpoint_dir = Path("checkpoints")
        if step is not None:
            checkpoint_path = checkpoint_dir / f"tier1_step_{step}.pt"
        else:
            checkpoints = list(checkpoint_dir.glob("tier1_step_*.pt"))
            if checkpoints:
                checkpoint_path = max(checkpoints, key=lambda p: int(p.stem.split("_")[2]))
            else:
                raise FileNotFoundError("No checkpoints found")

    print(f"Loading model from {checkpoint_path}")

    tokenizer = RustBPETokenizer.from_directory("tokenizer")
    vocab_size = tokenizer.get_vocab_size()
    image_token_id = vocab_size - 1

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

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    return model, tokenizer, image_token_id


def run_batch_loss_eval(model, tokenizer, image_token_id, dataset, data_dir, batch_size=4):
    """Run fast batched loss evaluation."""
    eval_dataset = EvalDataset(data_dir, dataset, tokenizer, image_token_id)

    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_token_id=0),
        num_workers=0,
        pin_memory=True,
    )

    device = next(model.parameters()).device
    results = []

    print(f"\n{Colors.BOLD}Running batched loss evaluation...{Colors.ENDC}")
    start_time = time.time()

    sample_idx = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(input_ids=input_ids, targets=targets, pixel_values=pixel_values, loss_reduction='none')
                mask = (targets != -1).float()
                per_sample_loss = (loss.view(targets.shape) * mask).sum(dim=1) / mask.sum(dim=1)

            for i, l in enumerate(per_sample_loss):
                if sample_idx < len(dataset):
                    sample = dataset[sample_idx]
                    loss_val = l.item()
                    results.append({"id": sample["id"], "loss": loss_val})

                    status = Colors.GREEN + "✓" if loss_val < 0.1 else Colors.RED + "✗"
                    print(f"  {sample['id']}: loss = {loss_val:.4f} {status}{Colors.ENDC}")
                    sample_idx += 1

    elapsed = time.time() - start_time
    avg_loss = sum(r["loss"] for r in results) / len(results)

    print(f"\n{Colors.BOLD}Batch eval completed in {elapsed:.2f}s{Colors.ENDC}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Throughput: {len(results)/elapsed:.1f} samples/sec")

    return results


def run_inference(model, tokenizer, image_path: str, prompt: str) -> str:
    """Run inference on a single image."""
    from PIL import Image
    from nanochat.image_process import process_image, count_vision_tokens

    image = Image.open(image_path).convert("RGB")
    pixel_values = process_image(image, base_size=1024)

    if torch.cuda.is_available():
        pixel_values = pixel_values.cuda()

    image_token_id = model.image_token_id
    num_vision_tokens = count_vision_tokens(base_size=1024)

    prompt_text = prompt.replace("<image>", "").strip()
    prompt_tokens = tokenizer.encode(prompt_text)

    tokens = [image_token_id] * num_vision_tokens + prompt_tokens
    input_ids = torch.tensor([tokens], dtype=torch.long)

    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=512,
            temperature=0.0,
        )

    generated_ids = output_ids[0, len(tokens):].tolist()
    return tokenizer.decode(generated_ids)


def run_batched_generation(model, tokenizer, image_token_id, dataset, data_dir, batch_size=2):
    """Run generation with batched vision encoding."""
    from PIL import Image
    from nanochat.image_process import process_image, count_vision_tokens

    num_vision_tokens = count_vision_tokens(base_size=1024)
    device = next(model.parameters()).device

    results = []
    print(f"\n{Colors.BOLD}Running batched generation (batch_size={batch_size})...{Colors.ENDC}")
    start_time = time.time()

    # Process in batches for vision encoding, but generate sequentially
    for batch_start in range(0, len(dataset), batch_size):
        batch_samples = dataset[batch_start:batch_start + batch_size]

        # Batch encode images
        pixel_values_list = []
        input_ids_list = []
        prompt_lens = []

        for sample in batch_samples:
            image_path = Path(data_dir) / sample["image"]
            image = Image.open(image_path).convert("RGB")
            pixel_values = process_image(image, base_size=1024)
            pixel_values_list.append(pixel_values.squeeze(0))

            prompt_text = sample["prompt"].replace("<image>", "").strip()
            prompt_tokens = tokenizer.encode(prompt_text)
            tokens = [image_token_id] * num_vision_tokens + prompt_tokens
            input_ids_list.append(torch.tensor(tokens, dtype=torch.long))
            prompt_lens.append(len(tokens))

        # Stack for batch vision encoding
        pixel_values_batch = torch.stack(pixel_values_list).to(device)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Pre-compute vision embeddings for entire batch
            vision_embeds_batch = model.encode_images(pixel_values_batch)

        # Generate for each sample using cached vision embeddings
        for i, sample in enumerate(batch_samples):
            input_ids = input_ids_list[i].unsqueeze(0).to(device)
            vision_embeds = vision_embeds_batch[i:i+1]

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                # Generate using cached vision embeddings
                generated_ids = input_ids.clone()
                for _ in range(512):
                    logits = model.forward(generated_ids, vision_embeds=vision_embeds)
                    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated_ids = torch.cat([generated_ids, next_id], dim=1)

                output_ids = generated_ids[0, prompt_lens[i]:].tolist()
                generated = tokenizer.decode(output_ids)

            results.append({
                "id": sample["id"],
                "expected": sample["answer"],
                "generated": generated,
            })

            # Print progress
            expected_words = set(sample["answer"].lower().split())
            generated_words = set(generated.lower().split())
            overlap = len(expected_words & generated_words) / max(len(expected_words), 1)

            status = Colors.GREEN + "✓" if overlap > 0.8 else Colors.YELLOW + "~" if overlap > 0.5 else Colors.RED + "✗"
            print(f"  {sample['id']}: {overlap:.0%} match {status}{Colors.ENDC}")

    elapsed = time.time() - start_time
    print(f"\n{Colors.BOLD}Generation completed in {elapsed:.2f}s{Colors.ENDC}")
    print(f"Throughput: {len(results)/elapsed:.2f} samples/sec")

    return results


def print_comparison(sample_id: str, sample_type: str, prompt: str,
                     expected: str, generated: str):
    """Print side-by-side comparison."""
    print(f"\n{'='*80}")
    print(f"{Colors.HEADER}{Colors.BOLD}[{sample_id}] {sample_type}{Colors.ENDC}")
    print(f"{Colors.BLUE}Prompt: {prompt[:60]}...{Colors.ENDC}" if len(prompt) > 60 else f"{Colors.BLUE}Prompt: {prompt}{Colors.ENDC}")
    print(f"{'='*80}")

    print(f"\n{Colors.GREEN}{Colors.BOLD}EXPECTED:{Colors.ENDC}")
    print("-" * 40)
    if len(expected) > 500:
        print(expected[:500] + f"\n... ({len(expected)} chars total)")
    else:
        print(expected)

    print(f"\n{Colors.YELLOW}{Colors.BOLD}GENERATED:{Colors.ENDC}")
    print("-" * 40)
    if generated:
        if len(generated) > 500:
            print(generated[:500] + f"\n... ({len(generated)} chars total)")
        else:
            print(generated)
    else:
        print(f"{Colors.RED}(no output){Colors.ENDC}")

    if generated:
        expected_words = set(expected.lower().split())
        generated_words = set(generated.lower().split())
        overlap = len(expected_words & generated_words) / max(len(expected_words), 1)

        if overlap > 0.8:
            print(f"\n{Colors.GREEN}✓ Good match ({overlap:.0%} word overlap){Colors.ENDC}")
        elif overlap > 0.5:
            print(f"\n{Colors.YELLOW}~ Partial match ({overlap:.0%} word overlap){Colors.ENDC}")
        else:
            print(f"\n{Colors.RED}✗ Poor match ({overlap:.0%} word overlap){Colors.ENDC}")


def main():
    parser = argparse.ArgumentParser(description="Vision sample sanity check")
    parser.add_argument("--step", type=int, help="Checkpoint step to load")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--mock", action="store_true", help="Run without model")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples")
    parser.add_argument("--batch-loss", action="store_true", help="Run fast batched loss eval only")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--verbose", action="store_true", help="Show full output comparisons")
    args = parser.parse_args()

    print(f"{Colors.BOLD}Loading dataset from {args.data_dir}/overfit_dataset.json{Colors.ENDC}")
    dataset = load_dataset(args.data_dir)
    print(f"Found {len(dataset)} samples")

    if args.samples:
        dataset = dataset[:args.samples]

    if args.mock:
        print(f"\n{Colors.YELLOW}Running in mock mode{Colors.ENDC}")
        for sample in dataset:
            print_comparison(sample["id"], sample["type"], sample["prompt"],
                           sample["answer"], sample["answer"])
        return

    try:
        model, tokenizer, image_token_id = load_model(args.checkpoint, args.step)
    except Exception as e:
        print(f"{Colors.RED}Error loading model: {e}{Colors.ENDC}")
        return

    if args.batch_loss:
        # Fast batched loss evaluation
        run_batch_loss_eval(model, tokenizer, image_token_id, dataset, args.data_dir, args.batch_size)
    else:
        # Full generation with output
        print(f"\n{Colors.BOLD}Running inference on {len(dataset)} samples...{Colors.ENDC}")

        for sample in dataset:
            image_path = Path(args.data_dir) / sample["image"]
            try:
                generated = run_inference(model, tokenizer, str(image_path), sample["prompt"])
            except Exception as e:
                generated = f"[ERROR: {e}]"

            print_comparison(sample["id"], sample["type"], sample["prompt"],
                           sample["answer"], generated)

    print(f"\n{'='*80}")
    print(f"{Colors.BOLD}Done! Processed {len(dataset)} samples.{Colors.ENDC}")


if __name__ == "__main__":
    main()
