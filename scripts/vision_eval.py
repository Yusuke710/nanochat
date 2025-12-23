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
from nanochat.checkpoint_manager import find_last_step, load_checkpoint

from tasks.fox import Fox
from tasks.omnidocbench import OmniDocBench

# -----------------------------------------------------------------------------
# config
task = "fox"  # fox | omnidocbench
checkpoint_dir = "checkpoints"  # checkpoint directory
step = -1  # step to load (-1 = auto-detect latest)
max_samples = -1  # -1 = all samples
# model config (will be loaded from metadata if available)
sequence_len = 4096
n_layer = 20
n_head = 16
n_kv_head = 16
n_embd = 1280
# -----------------------------------------------------------------------------

# Override from command line
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())

def main():
    global step

    # distributed / precision setup
    device_type = autodetect_device_type()
    _, ddp_rank, _, _, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # find checkpoint using checkpoint_manager
    if step < 0:
        step = find_last_step(checkpoint_dir)
    print0(f"Loading checkpoint from step {step}...")

    # load checkpoint and metadata
    model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)

    # get model config from metadata (fallback to defaults)
    model_config = meta_data.get("model_config", {})
    seq_len = model_config.get("sequence_len", sequence_len)
    vocab_size = model_config.get("vocab_size", 65540)
    num_layers = model_config.get("n_layer", n_layer)
    num_heads = model_config.get("n_head", n_head)
    num_kv_heads = model_config.get("n_kv_head", n_kv_head)
    model_dim = model_config.get("n_embd", n_embd)

    # build model
    tokenizer = RustBPETokenizer.from_directory("tokenizer")
    image_token_id = tokenizer.encode_special("<|image|>")
    gpt_config = GPTConfig(
        sequence_len=seq_len,
        vocab_size=vocab_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        n_embd=model_dim,
    )
    model = build_nano_deepseek_ocr(gpt_config=gpt_config)
    model.set_image_token_id(image_token_id)
    model.load_state_dict(model_data)
    model = model.eval().to(device)

    # load dataset
    print0(f"Loading {task} dataset...")
    dataset = Fox() if task == "fox" else OmniDocBench(lang="english")

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
