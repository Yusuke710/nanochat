"""
Vision Mid Training (Stage 2) - Train with SAM frozen.

Usage:
    python -m scripts.vis_mid_train                     # Train with defaults
    python -m scripts.vis_mid_train --steps=5000        # Custom steps
    python -m scripts.vis_mid_train --resume_step=1000  # Resume from checkpoint
    python -m scripts.vis_mid_train --text_ratio=0.1    # 10% text, 90% vision

Distributed:
    torchrun --nproc_per_node=8 -m scripts.vis_mid_train

Stage 2 differences from Stage 1:
    - SAM encoder frozen (only CLIP, projector, GPT trained)
    - StepLR scheduler instead of constant LR
    - Lower learning rate (3e-5)
    - Longer sequences (8192)
    - Mixed vision + text training (configurable ratio)
"""

import os
import time
import random
from contextlib import nullcontext

import torch

from nanochat.gpt import GPTConfig
from nanochat.nano_deepseek_ocr import build_nano_deepseek_ocr
from nanochat.deepencoder.load_pretrained import (
    load_sam_weights_from_hf,
    load_clip_weights_from_hf,
    load_nanochat_gpt_from_hf,
)
from nanochat.vision_dataloader import create_vision_loader
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type

# -----------------------------------------------------------------------------
# User settings (overridable via CLI)
run = "dummy"  # wandb run name ("dummy" = no wandb logging)
# Data
data_dir = "data"  # directory containing train.json, val.json, images/
text_ratio = 0.1  # fraction of steps using text-only data (0.0 = vision only)
# Model
base_size = 1024  # image resolution
seq_len = 8192  # longer sequences for stage 2
# Training
steps = 5000  # number of training steps
batch_size = 4  # batch size (smaller due to longer sequences)
lr = 3e-5  # learning rate (lower than stage 1)
weight_decay = 0.0  # weight decay
grad_clip = 1.0  # gradient clipping
warmup_steps = 100  # LR warmup steps
# LR schedule (StepLR)
lr_decay_step = 2000  # decay LR every N steps
lr_decay_gamma = 0.1  # multiply LR by this factor at each decay
# Checkpointing
checkpoint_dir = "checkpoints"  # where to save checkpoints
save_every = 500  # save every N steps (-1 = only at end)
resume_step = -1  # resume from step (-1 = fresh start)
resume_from_stage1 = ""  # path to stage 1 checkpoint to resume from
# Evaluation
eval_every = 100  # evaluate every N steps
eval_steps = 2  # number of batches to evaluate
# Runtime
device_type = ""  # cuda|cpu|mps (empty = autodetect)

# Override from command line
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

print0(f"Device: {device}, DDP world size: {ddp_world_size}")

# -----------------------------------------------------------------------------
# Load tokenizer
print0("Loading tokenizer...")
tokenizer = RustBPETokenizer.from_directory("tokenizer")
vocab_size = tokenizer.get_vocab_size()
image_token_id = tokenizer.encode_special("<|image|>")
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Build model (use pretrained vocab size, then expand for new tokens)
print0("Building model...")
pretrained_vocab_size = 65536  # original nanochat vocab size
num_layers = 20
model_dim = 1280
num_heads = 16
num_kv_heads = 16
gpt_config = GPTConfig(
    sequence_len=seq_len,
    vocab_size=pretrained_vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
)
model = build_nano_deepseek_ocr(gpt_config=gpt_config)

# Print model config
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")
print0(f"seq_len: {seq_len}")
print0(f"base_size: {base_size}")

# Load weights (either from stage 1 checkpoint or pretrained)
if resume_from_stage1:
    print0(f"Loading from stage 1 checkpoint: {resume_from_stage1}")
    # Expand embeddings first if needed
    if vocab_size > pretrained_vocab_size:
        old_wte = model.gpt.transformer.wte.weight.data
        old_lm_head = model.gpt.lm_head.weight.data
        model.gpt.transformer.wte = torch.nn.Embedding(vocab_size, gpt_config.n_embd)
        model.gpt.lm_head = torch.nn.Linear(gpt_config.n_embd, vocab_size, bias=False)
        model.gpt.transformer.wte.weight.data[:pretrained_vocab_size] = old_wte
        model.gpt.lm_head.weight.data[:pretrained_vocab_size] = old_lm_head
    model.load_state_dict(torch.load(resume_from_stage1, map_location="cpu", weights_only=False))
else:
    # Load pretrained weights
    print0("Loading pretrained weights...")
    hf_token = os.getenv("HF_TOKEN")
    model.sam_model = load_sam_weights_from_hf(model.sam_model, hf_token=hf_token, verbose=False)
    model.vision_model = load_clip_weights_from_hf(model.vision_model, hf_token=hf_token, verbose=False)
    model.gpt = load_nanochat_gpt_from_hf(model.gpt, hf_token=hf_token, verbose=False)

    # Expand embeddings for new special tokens
    if vocab_size > pretrained_vocab_size:
        print0(f"Expanding embeddings from {pretrained_vocab_size} to {vocab_size}...")
        old_wte = model.gpt.transformer.wte.weight.data
        old_lm_head = model.gpt.lm_head.weight.data
        model.gpt.transformer.wte = torch.nn.Embedding(vocab_size, gpt_config.n_embd)
        model.gpt.lm_head = torch.nn.Linear(gpt_config.n_embd, vocab_size, bias=False)
        model.gpt.transformer.wte.weight.data[:pretrained_vocab_size] = old_wte
        model.gpt.lm_head.weight.data[:pretrained_vocab_size] = old_lm_head
        model.gpt.transformer.wte.weight.data[pretrained_vocab_size:].normal_(mean=0.0, std=0.02)
        model.gpt.lm_head.weight.data[pretrained_vocab_size:].normal_(mean=0.0, std=0.02)

model.set_image_token_id(image_token_id)

# -----------------------------------------------------------------------------
# Freeze SAM encoder (Stage 2: only train CLIP, projector, GPT)
print0("Freezing SAM encoder...")
for p in model.sam_model.parameters():
    p.requires_grad = False

model = model.to(device)

# Model stats
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
tokens_per_batch = batch_size * seq_len * ddp_world_size
# Use GPT's estimate_flops for transformer part (Chinchilla formula)
num_flops_per_token = model.gpt.estimate_flops()
print0(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")
print0(f"Tokens per batch: {tokens_per_batch:,}")
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# H100 theoretical peak (bfloat16, no sparsity)
promised_flops_per_sec = 989e12 * ddp_world_size

# -----------------------------------------------------------------------------
# Resume from checkpoint if requested
start_step = 0
if resume_step >= 0:
    ckpt_path = os.path.join(checkpoint_dir, f"step_{resume_step}.pt")
    print0(f"Resuming from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    start_step = resume_step

# -----------------------------------------------------------------------------
# Setup optimizer (only trainable params)
trainable_params_list = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params_list, lr=lr, weight_decay=weight_decay)

def get_lr(step):
    """Warmup then StepLR decay."""
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    # StepLR: decay by gamma every lr_decay_step steps
    n_decays = (step - warmup_steps) // lr_decay_step
    return lr * (lr_decay_gamma ** n_decays)

# -----------------------------------------------------------------------------
# Setup dataloaders
print0(f"Loading vision data from {data_dir}/")
vision_loader = create_vision_loader(batch_size, seq_len, data_dir, tokenizer, "train", base_size)
val_loader_fn = lambda: create_vision_loader(batch_size, seq_len, data_dir, tokenizer, "val", base_size)

# Text data loader (FineWeb parquet files in ~/.cache/nanochat/base_data/)
text_loader = None
if text_ratio > 0:
    print0(f"Loading text data (text_ratio={text_ratio})")
    text_loader = tokenizing_distributed_data_loader(
        B=batch_size, T=seq_len, split="train", device=device,
    )

# Verify first batch
inputs, targets, pixel_values = next(iter(vision_loader))
print0(f"Batch shapes: inputs={inputs.shape}, targets={targets.shape}, pixel_values={pixel_values.shape}")

# -----------------------------------------------------------------------------
# Loop state
min_val_loss = float("inf")
smooth_train_loss = 0.0
total_training_time = 0.0

# -----------------------------------------------------------------------------
# Training loop
print0(f"\nStarting Stage 2 training for {steps} steps...")
model.train()
vision_iter = iter(vision_loader)

for step in range(start_step, steps):
    last_step = (step == steps - 1)

    # -------------------------------------------------------------------------
    # Evaluation
    if eval_every > 0 and (last_step or (step > 0 and step % eval_every == 0)):
        model.eval()
        val_loader = val_loader_fn()
        val_loss_sum = 0.0
        with torch.no_grad():
            for i, (val_inputs, val_targets, val_pv) in enumerate(val_loader):
                if i >= eval_steps:
                    break
                val_inputs = val_inputs.to(device, non_blocking=True)
                val_targets = val_targets.to(device, non_blocking=True)
                val_pv = val_pv.to(device, non_blocking=True)
                with autocast_ctx:
                    val_loss = model(input_ids=val_inputs, targets=val_targets, pixel_values=val_pv)
                val_loss_sum += val_loss.item()
        val_loss_avg = val_loss_sum / eval_steps
        if val_loss_avg < min_val_loss:
            min_val_loss = val_loss_avg
        print0(f"Step {step:05d} | Validation loss: {val_loss_avg:.4f} | min: {min_val_loss:.4f}")
        model.train()

    # -------------------------------------------------------------------------
    # Get next batch
    if text_loader is not None and random.random() < text_ratio:
        inputs, targets = next(text_loader)
        pixel_values = None
    else:
        try:
            inputs, targets, pixel_values = next(vision_iter)
        except StopIteration:
            vision_iter = iter(vision_loader)
            inputs, targets, pixel_values = next(vision_iter)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        pixel_values = pixel_values.to(device, non_blocking=True)

    # -------------------------------------------------------------------------
    # Training step with timing
    synchronize()
    t0 = time.time()

    # Update LR
    current_lr = get_lr(step)
    for group in optimizer.param_groups:
        group["lr"] = current_lr

    # Forward/backward
    optimizer.zero_grad()
    with autocast_ctx:
        loss = model(input_ids=inputs, targets=targets, pixel_values=pixel_values)
    loss.backward()

    # Gradient clipping
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(trainable_params_list, grad_clip)

    optimizer.step()

    synchronize()
    t1 = time.time()
    dt = t1 - t0

    # -------------------------------------------------------------------------
    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * loss.item()
    debiased_loss = smooth_train_loss / (1 - ema_beta ** (step - start_step + 1))

    # Timing stats
    if step > start_step + 5:
        total_training_time += dt
    tok_per_sec = int(tokens_per_batch / dt)
    flops_per_sec = num_flops_per_token * tokens_per_batch / dt
    mfu = 100 * flops_per_sec / promised_flops_per_sec

    pct_done = 100 * (step + 1) / steps
    print0(f"step {step:05d}/{steps} ({pct_done:05.2f}%) | loss: {debiased_loss:.4f} | lr: {current_lr:.2e} | "
           f"dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")

    # -------------------------------------------------------------------------
    # Checkpointing
    if save_every > 0 and (step + 1) % save_every == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, f"step_{step + 1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print0(f"Saved checkpoint to {ckpt_path}")

# -----------------------------------------------------------------------------
# Final checkpoint
os.makedirs(checkpoint_dir, exist_ok=True)
final_path = os.path.join(checkpoint_dir, f"step_{steps}.pt")
torch.save(model.state_dict(), final_path)
print0(f"\nFinal checkpoint saved to {final_path}")

# Final stats
print0(f"Total training time: {total_training_time / 60:.2f}m")
print0(f"Minimum val loss: {min_val_loss:.4f}")

if min_val_loss < 0.5:
    print0("SUCCESS: Stage 2 training converged!")

compute_cleanup()
