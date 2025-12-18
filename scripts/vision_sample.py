"""
Vision Sample - Sanity check script for nano-deepseek-ocr

Usage:
    python -m scripts.vision_sample                    # Run with latest checkpoint
    python -m scripts.vision_sample --step 1000        # Run with specific step
    python -m scripts.vision_sample --mock             # Test script without model

Loads 10 test images from data/overfit_dataset.json and shows EXPECTED vs GENERATED.
"""

import argparse
import json
import os
from pathlib import Path

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


def load_model(checkpoint_path: str = None, step: int = None):
    """
    Load nano-deepseek-ocr model.

    TODO: Implement once model is built
    """
    try:
        import torch
        from nanochat.nano_deepseek_ocr import NanoDeepseekOCR
        from transformers import AutoTokenizer

        # Determine checkpoint path
        if checkpoint_path is None:
            checkpoint_dir = Path("checkpoints")
            if step is not None:
                checkpoint_path = checkpoint_dir / f"step_{step}.pt"
            else:
                # Find latest checkpoint
                checkpoints = list(checkpoint_dir.glob("step_*.pt"))
                if checkpoints:
                    checkpoint_path = max(checkpoints, key=lambda p: int(p.stem.split("_")[1]))
                else:
                    raise FileNotFoundError("No checkpoints found")

        print(f"Loading model from {checkpoint_path}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("nanochat-students/base-d20")
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})

        # Load model
        model = NanoDeepseekOCR.from_pretrained(checkpoint_path)
        model = model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        return model, tokenizer

    except ImportError as e:
        print(f"{Colors.YELLOW}Warning: Could not import model ({e}){Colors.ENDC}")
        print(f"{Colors.YELLOW}Run with --mock to test without model{Colors.ENDC}")
        return None, None
    except FileNotFoundError as e:
        print(f"{Colors.YELLOW}Warning: {e}{Colors.ENDC}")
        print(f"{Colors.YELLOW}Run with --mock to test without model{Colors.ENDC}")
        return None, None


def run_inference(model, tokenizer, image_path: str, prompt: str) -> str:
    """
    Run inference on a single image.

    TODO: Implement once model is built
    """
    import torch
    from PIL import Image
    from nanochat.image_process import process_image

    # Load and process image
    image = Image.open(image_path).convert("RGB")
    pixel_values = process_image(image)

    if torch.cuda.is_available():
        pixel_values = pixel_values.cuda()

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=512,
            temperature=0.0,  # Greedy for reproducibility
        )

    # Decode
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Remove prompt from output
    if generated.startswith(prompt.replace("<image>", "").strip()):
        generated = generated[len(prompt.replace("<image>", "").strip()):].strip()

    return generated


def print_comparison(sample_id: str, sample_type: str, prompt: str,
                     expected: str, generated: str):
    """Print side-by-side comparison of expected vs generated."""
    print(f"\n{'='*80}")
    print(f"{Colors.HEADER}{Colors.BOLD}[{sample_id}] {sample_type}{Colors.ENDC}")
    print(f"{Colors.BLUE}Prompt: {prompt[:60]}...{Colors.ENDC}" if len(prompt) > 60 else f"{Colors.BLUE}Prompt: {prompt}{Colors.ENDC}")
    print(f"{'='*80}")

    print(f"\n{Colors.GREEN}{Colors.BOLD}EXPECTED:{Colors.ENDC}")
    print("-" * 40)
    # Show first 500 chars of expected
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

    # Simple similarity check
    if generated:
        # Check if key parts match (very basic)
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
    parser.add_argument("--mock", action="store_true", help="Run without model (show expected only)")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to run (default: all)")
    args = parser.parse_args()

    # Load dataset
    print(f"{Colors.BOLD}Loading dataset from {args.data_dir}/overfit_dataset.json{Colors.ENDC}")
    dataset = load_dataset(args.data_dir)
    print(f"Found {len(dataset)} samples")

    if args.samples:
        dataset = dataset[:args.samples]

    # Load model (or skip if mock)
    model, tokenizer = None, None
    if not args.mock:
        model, tokenizer = load_model(args.checkpoint, args.step)
        if model is None:
            print(f"\n{Colors.YELLOW}Falling back to mock mode{Colors.ENDC}")
            args.mock = True

    # Run inference on each sample
    print(f"\n{Colors.BOLD}Running inference on {len(dataset)} samples...{Colors.ENDC}")

    results = {"good": 0, "partial": 0, "poor": 0}

    for sample in dataset:
        image_path = Path(args.data_dir) / sample["image"]

        if args.mock:
            generated = "[MOCK MODE - no model loaded]"
        else:
            try:
                generated = run_inference(model, tokenizer, str(image_path), sample["prompt"])
            except Exception as e:
                generated = f"[ERROR: {e}]"

        print_comparison(
            sample_id=sample["id"],
            sample_type=sample["type"],
            prompt=sample["prompt"],
            expected=sample["answer"],
            generated=generated if not args.mock else sample["answer"]  # In mock mode, show expected as "generated" for testing
        )

    print(f"\n{'='*80}")
    print(f"{Colors.BOLD}Done! Processed {len(dataset)} samples.{Colors.ENDC}")

    if args.mock:
        print(f"\n{Colors.YELLOW}Note: Running in mock mode. Use a trained model for real inference.{Colors.ENDC}")


if __name__ == "__main__":
    main()
