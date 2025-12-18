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
from nanochat.vision_dataloader import vision_data_loader
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
log_every = 10  # log training loss every N steps
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
print0(f"Vocab size: {vocab_size}, image token ID: {image_token_id}")

# -----------------------------------------------------------------------------
# Build model (use pretrained vocab size, then expand for new tokens)
print0("Building model...")
pretrained_vocab_size = 65536  # original nanochat vocab size
seq_len = 8192  # longer sequences for stage 2
gpt_config = GPTConfig(
    sequence_len=seq_len,
    vocab_size=pretrained_vocab_size,
    n_layer=20,
    n_head=16,
    n_kv_head=16,
    n_embd=1280,
)
model = build_nano_deepseek_ocr(gpt_config=gpt_config)

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
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print0(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")

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
# Setup dataloaders (Karpathy style: vision loader + text loader + val loader builder)
print0(f"Loading vision data from {data_dir}/")
vision_loader = vision_data_loader(
    B=batch_size, T=seq_len, data_dir=data_dir, device=device,
    tokenizer=tokenizer, split="train", base_size=base_size,
)
build_val_loader = lambda: vision_data_loader(
    B=batch_size, T=seq_len, data_dir=data_dir, device=device,
    tokenizer=tokenizer, split="val", base_size=base_size,
)

# Text data loader (FineWeb parquet files in ~/.cache/nanochat/base_data/)
text_loader = None
if text_ratio > 0:
    print0(f"Loading text data (text_ratio={text_ratio})")
    text_loader = tokenizing_distributed_data_loader(
        B=batch_size, T=seq_len, split="train", device=device,
    )

# Verify first batch
inputs, targets, pixel_values = next(vision_loader)
print0(f"Batch shapes: inputs={inputs.shape}, targets={targets.shape}, pixel_values={pixel_values.shape}")

# Recreate after verification
vision_loader = vision_data_loader(
    B=batch_size, T=seq_len, data_dir=data_dir, device=device,
    tokenizer=tokenizer, split="train", base_size=base_size,
)

# -----------------------------------------------------------------------------
# Loop state
min_val_loss = float("inf")
smooth_train_loss = 0.0

# -----------------------------------------------------------------------------
# Training loop
print0(f"\nStarting Stage 2 training for {steps} steps...")
model.train()
start_time = time.time()

for step in range(start_step, steps):
    last_step = (step == steps - 1)

    # -------------------------------------------------------------------------
    # Evaluation (Karpathy style: fresh val loader, multiple eval steps)
    if eval_every > 0 and (last_step or (step > 0 and step % eval_every == 0)):
        model.eval()
        val_loader = build_val_loader()
        val_loss_sum = 0.0
        with torch.no_grad():
            for _ in range(eval_steps):
                val_inputs, val_targets, val_pv = next(val_loader)
                with autocast_ctx:
                    val_loss = model(input_ids=val_inputs, targets=val_targets, pixel_values=val_pv)
                val_loss_sum += val_loss.item()
        val_loss_avg = val_loss_sum / eval_steps
        if val_loss_avg < min_val_loss:
            min_val_loss = val_loss_avg
        print0(f"step {step:05d} | val loss: {val_loss_avg:.4f} | min: {min_val_loss:.4f}")
        model.train()

    # -------------------------------------------------------------------------
    # Training step - pick vision or text data
    if text_loader is not None and random.random() < text_ratio:
        inputs, targets = next(text_loader)
        pixel_values = None
    else:
        inputs, targets, pixel_values = next(vision_loader)

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

    # -------------------------------------------------------------------------
    # Logging (EMA smoothed like Karpathy)
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * loss.item()
    debiased_loss = smooth_train_loss / (1 - ema_beta ** (step - start_step + 1))

    if (step + 1) % log_every == 0:
        elapsed = time.time() - start_time
        samples_per_sec = ((step - start_step + 1) * batch_size) / elapsed
        print0(f"step {step + 1:05d}/{steps} | loss: {debiased_loss:.4f} | lr: {current_lr:.2e} | "
               f"{samples_per_sec:.1f} samples/s")

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
total_time = time.time() - start_time
print0(f"Total training time: {total_time / 60:.2f}m")
print0(f"Minimum val loss: {min_val_loss:.4f}")

if min_val_loss < 0.5:
    print0("SUCCESS: Stage 2 training converged!")

compute_cleanup()
