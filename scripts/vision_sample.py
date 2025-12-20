"""
Vision Sample - Evaluation script for nano-deepseek-ocr.

Usage:
    python -m scripts.vision_sample --step 150
    python -m scripts.vision_sample --step 150 --loss-only
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from nanochat.gpt import GPTConfig
from nanochat.nano_deepseek_ocr import build_nano_deepseek_ocr
from nanochat.tokenizer import RustBPETokenizer
from nanochat.image_process import process_image, count_vision_tokens, expand_image_tokens
from nanochat.engine import Engine

GREEN, YELLOW, RED, BOLD, END = '\033[92m', '\033[93m', '\033[91m', '\033[1m', '\033[0m'


def load_model(checkpoint_path):
    """Load model from checkpoint."""
    tokenizer = RustBPETokenizer.from_directory("tokenizer")
    image_token_id = tokenizer.encode_special("<|image|>")

    gpt_config = GPTConfig(sequence_len=4096, vocab_size=tokenizer.get_vocab_size(),
                           n_layer=20, n_head=16, n_kv_head=16, n_embd=1280)
    model = build_nano_deepseek_ocr(gpt_config=gpt_config)
    model.set_image_token_id(image_token_id)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.eval().to(device), tokenizer, image_token_id, device


def evaluate_sample(model, tokenizer, image_token_id, sample, data_dir, device, generate=True):
    """Evaluate single sample: compute loss and optionally generate."""
    n_img_tokens = count_vision_tokens(base_size=1024)

    # Process image
    pixel_values = process_image(
        Image.open(Path(data_dir) / sample["image"]).convert("RGB"), base_size=1024
    ).unsqueeze(0).to(device)

    # Special tokens for conversation structure (matching chat_cli.py)
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    # Tokenize prompt with conversation structure
    prompt = sample["prompt"].replace("<image>", "<|image|>")
    prompt_content = tokenizer.enc.encode(prompt, allowed_special={"<|image|>"})
    # Wrap: <bos><|user_start|>prompt<|user_end|><|assistant_start|>
    prompt_ids = [bos, user_start] + list(prompt_content) + [user_end, assistant_start]
    expanded = expand_image_tokens(prompt_ids, image_token_id, n_img_tokens)
    # Answer with <|assistant_end|> for accurate loss calculation
    answer_ids = tokenizer.encode(sample["answer"]) + [assistant_end]
    prompt_len = len(expanded)

    # Compute loss
    full_ids = expanded + answer_ids
    input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)
    targets[:, :prompt_len - 1] = -1
    targets[input_ids == image_token_id] = -1

    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        loss = model(input_ids=input_ids, targets=targets, pixel_values=pixel_values).item()

    # Generate using Engine with KV cache
    generated = ""
    if generate:
        engine = Engine(model, tokenizer)
        gen_tokens = []
        with torch.autocast(device, dtype=torch.bfloat16):
            for token_column, _ in engine.generate(expanded, pixel_values=pixel_values,
                                                    max_tokens=512, temperature=0.0):
                token = token_column[0]
                if token == assistant_end:
                    break
                gen_tokens.append(token)
        generated = tokenizer.decode(gen_tokens).strip()

    return loss, generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--step", type=int)
    parser.add_argument("--data-dir", type=str, default="data/overfit_samples")
    parser.add_argument("--dataset", type=str, default="val.json", help="Dataset file (default: val.json)")
    parser.add_argument("--loss-only", action="store_true")
    args = parser.parse_args()

    # Checkpoint path
    if args.checkpoint:
        ckpt = args.checkpoint
    elif args.step:
        ckpt = f"checkpoints/step_{args.step}.pt"
    else:
        ckpts = list(Path("checkpoints").glob("step_*.pt"))
        ckpt = str(max(ckpts, key=lambda p: int(p.stem.split("_")[1]))) if ckpts else None
    if not ckpt or not Path(ckpt).exists():
        print(f"{RED}Checkpoint not found{END}")
        return

    step_num = Path(ckpt).stem.split("_")[-1]
    output_path = f"{args.data_dir}/inference_results_step{step_num}.json"

    # Load dataset (val.json by default, or specify with --dataset)
    dataset_path = f"{args.data_dir}/{args.dataset}"
    with open(dataset_path, encoding="utf-8") as f:
        samples = json.load(f)
    print(f"Loading {ckpt}")
    model, tokenizer, image_token_id, device = load_model(ckpt)

    # Evaluate
    results, total_loss = [], 0.0
    print(f"\n{'ID':<15} {'Loss':>8} {'Overlap':>8}")
    print("=" * 40)

    for sample in samples:
        loss, generated = evaluate_sample(
            model, tokenizer, image_token_id, sample, args.data_dir, device, not args.loss_only
        )
        total_loss += loss

        exp_words = set(sample["answer"].lower().split())
        gen_words = set(generated.lower().split()) if generated else set()
        overlap = len(exp_words & gen_words) / max(len(exp_words), 1)

        status = GREEN + "✓" if loss < 0.1 else (YELLOW + "~" if overlap > 0.8 else RED + "✗")
        print(f"{sample['id']:<15} {loss:>8.4f} {overlap:>7.0%}  {status}{END}")

        results.append({
            "id": sample["id"], "type": sample["type"],
            "expected": sample["answer"], "generated": generated,
            "loss": round(loss, 6), "word_overlap": round(overlap, 3),
        })

    # Summary
    print("=" * 40)
    print(f"{BOLD}Avg loss: {total_loss/len(samples):.4f}{END}")
    if not args.loss_only:
        print(f"Avg overlap: {sum(r['word_overlap'] for r in results)/len(results):.1%}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
