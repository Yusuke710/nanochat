"""
Evaluate vision model on OCR benchmarks (Fox, OmniDocBench).

Usage:
    python -m scripts.vision_eval fox
    python -m scripts.vision_eval omnidocbench
    python -m scripts.vision_eval omnidocbench checkpoints/step_1000.pt 100
"""
import os
import time
from contextlib import nullcontext

import torch

from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type
from nanochat.gpt import GPTConfig
from nanochat.nano_deepseek_ocr import build_nano_deepseek_ocr
from nanochat.tokenizer import RustBPETokenizer
from nanochat.vision_eval import evaluate_vision_task

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


def main():
    import sys
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

    # load dataset
    print0(f"Loading {task} dataset...")
    dataset = Fox() if task == "fox" else OmniDocBench()

    # evaluate
    print0(f"Evaluating {len(dataset)} samples...")
    t0 = time.time()
    with autocast_ctx:
        out = evaluate_vision_task(model, tokenizer, dataset, device, max_samples=max_samples)
    dt = time.time() - t0

    # print results
    print0(f"\n{'='*60}")
    print0(f"RESULTS ({task})")
    print0(f"{'='*60}")
    print0(f"Score: {out['avg_score']:.4f}")
    print0(f"Samples: {out['num_samples']}")
    print0(f"Time: {dt:.1f}s ({out['num_samples']/dt:.1f} samples/s)")
    print0(f"{'='*60}")

    compute_cleanup()


if __name__ == "__main__":
    main()
