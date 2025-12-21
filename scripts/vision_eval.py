"""
Evaluate vision model on OCR benchmarks.

Usage:
    python -m scripts.vision_eval --task fox --step 500
    python -m scripts.vision_eval --task omnidocbench --step 500 --max-samples 100
"""

import argparse
import json
import time
from pathlib import Path

import torch

from nanochat.gpt import GPTConfig
from nanochat.nano_deepseek_ocr import build_nano_deepseek_ocr
from nanochat.tokenizer import RustBPETokenizer
from nanochat.vision_eval import evaluate_ocr

from tasks.fox import Fox
from tasks.omnidocbench import OmniDocBench

# -----------------------------------------------------------------------------

def evaluate_model(model, tokenizer, device, task_name, max_samples=-1):
    """
    Evaluate vision model on OCR benchmark.
    Fox: precision (per DeepSeek-OCR paper)
    OmniDocBench: normalized edit distance (per CVPR 2025 paper)
    """
    print(f"Loading {task_name} dataset...")
    if task_name == "fox":
        dataset = Fox()
    elif task_name == "omnidocbench":
        dataset = OmniDocBench()
    else:
        raise ValueError(f"Unknown task: {task_name}")

    print(f"Evaluating {len(dataset)} samples...")
    t0 = time.time()
    out = evaluate_ocr(model, tokenizer, dataset, device, max_samples=max_samples)
    dt = time.time() - t0

    # Primary metric depends on benchmark
    if task_name == "fox":
        primary = out["avg_precision"]
        print(f"Precision: {primary:.4f} | F1: {out['avg_f1']:.4f} | time: {dt:.1f}s")
    else:
        primary = 1 - out["avg_ned"]  # invert so higher is better
        print(f"NED: {out['avg_ned']:.4f} | F1: {out['avg_f1']:.4f} | time: {dt:.1f}s")

    return out

# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="fox", help="fox|omnidocbench")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--step", type=int)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    # Find checkpoint
    if args.checkpoint:
        ckpt = args.checkpoint
    elif args.step:
        ckpt = f"checkpoints/step_{args.step}.pt"
    else:
        ckpts = list(Path("checkpoints").glob("step_*.pt"))
        ckpt = str(max(ckpts, key=lambda p: int(p.stem.split("_")[1]))) if ckpts else None
    if not ckpt or not Path(ckpt).exists():
        print("Checkpoint not found")
        return

    # Load model
    print(f"Loading {ckpt}...")
    tokenizer = RustBPETokenizer.from_directory("tokenizer")
    image_token_id = tokenizer.encode_special("<|image|>")
    gpt_config = GPTConfig(sequence_len=4096, vocab_size=tokenizer.get_vocab_size(),
                           n_layer=20, n_head=16, n_kv_head=16, n_embd=1280)
    model = build_nano_deepseek_ocr(gpt_config=gpt_config)
    model.set_image_token_id(image_token_id)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(device)

    # Evaluate
    out = evaluate_model(model, tokenizer, device, args.task, args.max_samples)

    # Save
    Path(args.output_dir).mkdir(exist_ok=True)
    step_num = Path(ckpt).stem.split("_")[-1]
    output_path = f"{args.output_dir}/{args.task}_step{step_num}.json"
    with open(output_path, "w") as f:
        json.dump({"checkpoint": ckpt, **out}, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
