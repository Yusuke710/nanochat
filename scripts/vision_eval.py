"""
Evaluate vision model on OCR benchmarks (Fox, OmniDocBench).

Run on a single GPU:
python -m scripts.vision_eval

Run with torchrun on e.g. 2 GPUs:
torchrun --nproc_per_node=2 -m scripts.vision_eval

Metrics per DeepSeek-OCR paper:
- Fox: Precision (word-level, higher is better)
- OmniDocBench: Edit Distance (character-level, lower is better)
"""
import os
import csv
import time
from contextlib import nullcontext

import torch

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type
from nanochat.gpt import GPTConfig
from nanochat.nano_deepseek_ocr import build_nano_deepseek_ocr
from nanochat.tokenizer import RustBPETokenizer
from nanochat.vision_eval import evaluate_ocr

from tasks.fox import Fox
from tasks.omnidocbench import OmniDocBench

# -----------------------------------------------------------------------------
# config
task = "fox"  # fox | omnidocbench
checkpoint = ""  # path to checkpoint, empty = auto-detect latest
max_samples = -1  # -1 = all samples
# model config
sequence_len = 4096
n_layer = 20
n_head = 16
n_kv_head = 16
n_embd = 1280
# -----------------------------------------------------------------------------

def evaluate_model(model, tokenizer, device, task_name, max_samples=-1):
    """
    Evaluate vision model on OCR benchmark.
    Returns dict with metrics matching DeepSeek-OCR paper:
    - Fox: precision (word-level, higher is better)
    - OmniDocBench: ned (character-level, lower is better)
    """
    print0(f"Loading {task_name} dataset...")
    if task_name == "fox":
        dataset = Fox()
    elif task_name == "omnidocbench":
        dataset = OmniDocBench()
    else:
        raise ValueError(f"Unknown task: {task_name}")

    print0(f"Evaluating {len(dataset)} samples...")
    t0 = time.time()
    out = evaluate_ocr(model, tokenizer, dataset, device, max_samples=max_samples, verbose=False)
    dt = time.time() - t0

    # Primary metric per DeepSeek-OCR paper
    if task_name == "fox":
        primary_name = "Precision"
        primary_value = out["avg_precision"]
    else:
        primary_name = "Edit Distance"
        primary_value = out["avg_ned"]

    print0(f"\n{'='*60}")
    print0(f"RESULTS ({task_name})")
    print0(f"{'='*60}")
    print0(f"{primary_name}: {primary_value:.4f}")
    print0(f"{'='*60}")
    print0(f"All metrics:")
    print0(f"  Precision: {out['avg_precision']:.4f}")
    print0(f"  Recall:    {out['avg_f1']:.4f}")  # F1 used as proxy
    print0(f"  NED:       {out['avg_ned']:.4f}")
    print0(f"  Samples:   {out['num_samples']}")
    print0(f"  Time:      {dt:.1f}s ({out['num_samples']/dt:.1f} samples/s)")
    print0(f"{'='*60}")

    return {
        "task": task_name,
        "primary_metric": primary_name,
        "primary_value": primary_value,
        "precision": out["avg_precision"],
        "f1": out["avg_f1"],
        "ned": out["avg_ned"],
        "num_samples": out["num_samples"],
        "time": dt,
    }

# -----------------------------------------------------------------------------

def main():
    import sys
    # simple arg override: python -m scripts.vision_eval fox 500
    global task, checkpoint, max_samples
    if len(sys.argv) > 1:
        task = sys.argv[1]
    if len(sys.argv) > 2:
        if sys.argv[2].isdigit():
            checkpoint = f"checkpoints/step_{sys.argv[2]}.pt"
        else:
            checkpoint = sys.argv[2]
    if len(sys.argv) > 3:
        max_samples = int(sys.argv[3])

    # distributed / precision setup
    device_type = autodetect_device_type()
    _, ddp_rank, _, _, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # find checkpoint
    ckpt = checkpoint
    if not ckpt:
        from pathlib import Path
        ckpts = list(Path("checkpoints").glob("step_*.pt"))
        ckpt = str(max(ckpts, key=lambda p: int(p.stem.split("_")[1]))) if ckpts else None
    if not ckpt or not os.path.exists(ckpt):
        print0("Checkpoint not found")
        compute_cleanup()
        return

    # load model
    print0(f"Loading {ckpt}...")
    tokenizer = RustBPETokenizer.from_directory("tokenizer")
    image_token_id = tokenizer.encode_special("<|image|>")
    gpt_config = GPTConfig(
        sequence_len=sequence_len,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=n_embd,
    )
    model = build_nano_deepseek_ocr(gpt_config=gpt_config)
    model.set_image_token_id(image_token_id)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    model = model.eval().to(device)

    # evaluate
    with autocast_ctx:
        out = evaluate_model(model, tokenizer, device, task, max_samples)

    # save results (rank 0 only)
    if ddp_rank == 0:
        base_dir = get_base_dir()
        output_dir = os.path.join(base_dir, "vision_eval")
        os.makedirs(output_dir, exist_ok=True)
        step_num = os.path.basename(ckpt).replace("step_", "").replace(".pt", "")
        output_csv = os.path.join(output_dir, f"{task}_step{step_num}.csv")
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            writer.writerow([out["primary_metric"], f"{out['primary_value']:.4f}"])
            writer.writerow(["Precision", f"{out['precision']:.4f}"])
            writer.writerow(["F1", f"{out['f1']:.4f}"])
            writer.writerow(["NED", f"{out['ned']:.4f}"])
            writer.writerow(["Samples", out["num_samples"]])
        print0(f"Saved to {output_csv}")

    compute_cleanup()


if __name__ == "__main__":
    main()
