"""
Vision Token Training - Train vision encoder with LLM on vision data.

Usage:
    python -m scripts.vis_tok_train                     # Train with defaults
    python -m scripts.vis_tok_train --steps=500         # Custom steps

Distributed:
    torchrun --nproc_per_node=8 -m scripts.vis_tok_train
"""

import math
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist

from nanochat.gpt import GPTConfig
from nanochat.nano_deepseek_ocr import build_nano_deepseek_ocr, VisionConfig
from nanochat.deepencoder.load_pretrained import (
    load_sam_weights_from_hf,
    load_clip_weights_from_hf,
    load_nanochat_gpt_from_hf,
)
from nanochat.multimodal_dataloader import create_multimodal_loader
from nanochat.tokenizer import RustBPETokenizer
from tasks.finevision import FineVision
from tasks.common import TaskMixture
from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type, DummyWandb
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.vision_eval import evaluate_vision_task
import wandb

# -----------------------------------------------------------------------------
# User settings (overridable via CLI)
run = "dummy"  # wandb run name ("dummy" = no wandb logging)
# Model
base_size = 1024  # image resolution
seq_len = 4096  # sequence length
# Training
num_epochs = 1  # number of epochs (used if steps == -1)
steps = -1  # number of training steps (-1 = derive from num_epochs)
total_batch_size = 128  # effective batch size (1/10 of DeepSeek-OCR stage 1)
micro_batch_size = 8  # batch size per GPU per micro step
lr = 5e-5  # learning rate
weight_decay = 0.0  # weight decay
grad_clip = 1.0  # gradient clipping
warmup_steps = 10  # LR warmup steps
# Checkpointing
checkpoint_dir = "checkpoints"  # where to save checkpoints
save_every = -1  # save every N steps (-1 = only at end)
resume_step = -1  # resume from step (-1 = fresh start)
# Evaluation
eval_every = 50  # evaluate every N steps
eval_steps = 1  # number of batches to evaluate
eval_metrics_every = 100  # evaluate Fox/OmniDocBench metrics every N steps (-1 = disable)
eval_metrics_max_problems = 50  # max problems per task (Fox=112, OmniDocBench=1355)
# Runtime
device_type = ""  # cuda|cpu|mps (empty = autodetect)
sam_gradient_checkpointing = False  # save memory on SAM global attention (slower)

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

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nano-deepseekocr-stage1", name=run, config=user_config)

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
vision_config = VisionConfig(sam_gradient_checkpointing=sam_gradient_checkpointing)
model = build_nano_deepseek_ocr(gpt_config=gpt_config, vision_config=vision_config)

# Print model config
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")
print0(f"seq_len: {seq_len}")
print0(f"base_size: {base_size}")
print0(f"sam_gradient_checkpointing: {sam_gradient_checkpointing}")

# Load pretrained weights
print0("Loading pretrained weights...")
hf_token = os.getenv("HF_TOKEN")
model.sam_model = load_sam_weights_from_hf(model.sam_model, hf_token=hf_token, verbose=False)
model.vision_model = load_clip_weights_from_hf(model.vision_model, hf_token=hf_token, verbose=False)
model.gpt = load_nanochat_gpt_from_hf(model.gpt, hf_token=hf_token, verbose=False)

# Expand embeddings for new special tokens (e.g., <|image|>)
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
model = model.to(device)

# -----------------------------------------------------------------------------
# Resume from checkpoint if requested (BEFORE DDP wrapping)
start_step = 0
if resume_step >= 0:
    ckpt_path = os.path.join(checkpoint_dir, f"step_{resume_step}.pt")
    print0(f"Resuming from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    start_step = resume_step

# -----------------------------------------------------------------------------
# Wrap with DDP for multi-GPU training
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model  # unwrapped model for saving/attributes

# Model stats
num_params = sum(p.numel() for p in model.parameters())
tokens_per_batch = micro_batch_size * seq_len * ddp_world_size
# Use GPT's estimate_flops for transformer part (Chinchilla formula)
num_flops_per_token = raw_model.gpt.estimate_flops()
print0(f"Number of parameters: {num_params:,}")
print0(f"Tokens per batch: {tokens_per_batch:,}")
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# H100 theoretical peak (bfloat16, no sparsity)
promised_flops_per_sec = 989e12 * ddp_world_size

# -----------------------------------------------------------------------------
# Setup optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def get_lr(step):
    """Warmup then cosine annealing to 0."""
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, steps - warmup_steps)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))

# -----------------------------------------------------------------------------
# Setup dataloaders (unified multimodal pipeline with PyTorch DataLoader)
# Stage 1: OCR-focused training with olmOCR documents and books
# Note: FineVision uses start/stop to avoid train/val overlap (only has train split on HF)
# Val ratio ~5.2% to match Stage 2 text task proportions
# DDP: Rank 0 downloads datasets first, others wait then load from cache
if ddp_rank == 0:
    train_ds = TaskMixture([
        FineVision("olmOCR-mix-0225-documents", prompt="Free OCR.", start=12000),  # 229K PDF documents
        FineVision("olmOCR-mix-0225-books", prompt="Free OCR.", start=800),        # 15.2K book pages
    ])
    val_ds = TaskMixture([
        FineVision("olmOCR-mix-0225-documents", prompt="Free OCR.", stop=12000),   # 12K val samples
        FineVision("olmOCR-mix-0225-books", prompt="Free OCR.", stop=800),         # 800 val samples
    ])
if ddp:
    dist.barrier()
if ddp_rank != 0:
    train_ds = TaskMixture([
        FineVision("olmOCR-mix-0225-documents", prompt="Free OCR.", start=12000),
        FineVision("olmOCR-mix-0225-books", prompt="Free OCR.", start=800),
    ])
    val_ds = TaskMixture([
        FineVision("olmOCR-mix-0225-documents", prompt="Free OCR.", stop=12000),
        FineVision("olmOCR-mix-0225-books", prompt="Free OCR.", stop=800),
    ])
train_task_names = [t.__class__.__name__ for t in train_ds.tasks]
val_task_names = [t.__class__.__name__ for t in val_ds.tasks]
print0(f"Train tasks: {train_task_names}, Val tasks: {val_task_names}")
print0(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

# Derive gradient accumulation steps from total_batch_size (Karpathy nanoGPT pattern)
assert total_batch_size % (micro_batch_size * ddp_world_size) == 0, "total_batch_size must be divisible by micro_batch_size * ddp_world_size"
grad_accum_steps = total_batch_size // (micro_batch_size * ddp_world_size)
print0(f"Total batch size: {total_batch_size}")
print0(f"Micro batch size: {micro_batch_size}")
print0(f"Gradient accumulation steps: {grad_accum_steps}")
if steps == -1:
    # derive steps from num_epochs and the size of the dataset
    assert num_epochs > 0, "num_epochs must be positive if steps is -1"
    steps = (len(train_ds) // total_batch_size) * num_epochs
print0(f"Total steps: {steps}")

train_loader = create_multimodal_loader(train_ds, tokenizer, micro_batch_size, seq_len, base_size)
val_loader_fn = lambda: create_multimodal_loader(val_ds, tokenizer, micro_batch_size, seq_len, base_size)

# Verify first batch
inputs, targets, media = next(iter(train_loader))
pixel_values = media["pixel_values"]
print0(f"Batch shapes: inputs={inputs.shape}, targets={targets.shape}, pixel_values={pixel_values.shape}")

# -----------------------------------------------------------------------------
# Loop state
min_val_loss = float("inf")
smooth_train_loss = 0.0
total_training_time = 0.0

# -----------------------------------------------------------------------------
# Training loop
print0(f"\nStarting training for {steps} steps...")
model.train()
train_iter = iter(train_loader)

for step in range(start_step, steps):
    last_step = (step == steps - 1)

    # -------------------------------------------------------------------------
    # Evaluation
    if eval_every > 0 and (last_step or (step > 0 and step % eval_every == 0)):
        model.eval()
        val_loader = iter(val_loader_fn())
        losses = []
        for _ in range(eval_steps):
            val_inputs, val_targets, val_media = next(val_loader)
            val_inputs = val_inputs.to(device, non_blocking=True)
            val_targets = val_targets.to(device, non_blocking=True)
            val_pv = val_media["pixel_values"]
            if val_pv is not None:
                val_pv = val_pv.to(device, non_blocking=True)
            with torch.no_grad(), autocast_ctx:
                loss = model(input_ids=val_inputs, targets=val_targets, pixel_values=val_pv)
            losses.append(loss)
        val_loss = torch.stack(losses).mean()
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss = val_loss.item()
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        print0(f"Step {step:05d} | Validation loss: {val_loss:.4f} | min: {min_val_loss:.4f}")
        total_training_flops = num_flops_per_token * tokens_per_batch * (step + 1)
        wandb_run.log({"step": step, "total_training_flops": total_training_flops, "val/loss": val_loss})
        model.train()

    # -------------------------------------------------------------------------
    # Vision metrics (OmniDocBench NED ↓) per DeepSeek-OCR paper
    if eval_metrics_every > 0 and (last_step or (step > 0 and step % eval_metrics_every == 0)):
        from tasks.omnidocbench import OmniDocBench
        model.eval()
        with torch.no_grad(), autocast_ctx:
            omnidoc = OmniDocBench(lang="english")
            out = evaluate_vision_task(raw_model, tokenizer, omnidoc, device, max_samples=eval_metrics_max_problems)
        # avg_score is 1-NED, so NED = 1 - avg_score
        metrics = {"omnidoc_ned": round(1 - out["avg_score"], 4)}
        metrics_str = ', '.join(f'{k}: {v:.4f}' for k, v in metrics.items())
        print0(f"Step {step:05d} | {metrics_str}")
        wandb_run.log({"step": step, **metrics})
        model.train()

    # -------------------------------------------------------------------------
    # Training step with timing
    synchronize()
    t0 = time.time()

    # Update LR
    current_lr = get_lr(step)
    for group in optimizer.param_groups:
        group["lr"] = current_lr

    # Gradient accumulation loop
    optimizer.zero_grad()
    for micro_step in range(grad_accum_steps):
        # Get next batch
        try:
            inputs, targets, media = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs, targets, media = next(train_iter)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        pixel_values = media["pixel_values"]
        if pixel_values is not None:
            pixel_values = pixel_values.to(device, non_blocking=True)

        # Forward/backward
        with autocast_ctx:
            loss = model(input_ids=inputs, targets=targets, pixel_values=pixel_values)
        train_loss = loss.detach()  # for logging (last micro-step)
        loss = loss / grad_accum_steps  # normalize for gradient accumulation
        loss.backward()

    # Gradient clipping
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    synchronize()
    t1 = time.time()
    dt = t1 - t0

    # -------------------------------------------------------------------------
    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
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
    total_training_flops = num_flops_per_token * tokens_per_batch * (step + 1)
    wandb_run.log({
        "step": step,
        "train/loss": debiased_loss,
        "train/lr": current_lr,
        "train/dt": dt,
        "train/tok_per_sec": tok_per_sec,
        "train/mfu": mfu,
        "train/total_flops": total_training_flops,
    })

    # -------------------------------------------------------------------------
    # Checkpointing (master only)
    if master_process and save_every > 0 and (step + 1) % save_every == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, f"step_{step + 1}.pt")
        torch.save(raw_model.state_dict(), ckpt_path)
        print0(f"Saved checkpoint to {ckpt_path}")

# -----------------------------------------------------------------------------
# Final checkpoint (master only)
if master_process:
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save full model using checkpoint_manager pattern (model_{step:06d}.pt)
    meta_data = {
        "model_config": {
            "sequence_len": seq_len,
            "vocab_size": vocab_size,
            "n_layer": num_layers,
            "n_head": num_heads,
            "n_kv_head": num_kv_heads,
            "n_embd": model_dim,
        },
        "stage": "vis_tok_train",
        "num_epochs": num_epochs,
        "steps": steps,
        "base_size": base_size,
    }
    save_checkpoint(checkpoint_dir, steps, raw_model.state_dict(), None, meta_data, rank=ddp_rank)

    # Save DeepEncoder only (for Stage 2 - discard decoder per DeepSeek-OCR paper)
    deepencoder_state = {k: v for k, v in raw_model.state_dict().items() if not k.startswith('gpt.')}
    deepencoder_path = os.path.join(checkpoint_dir, "deepencoder_stage1.pt")
    torch.save(deepencoder_state, deepencoder_path)
    print0(f"DeepEncoder checkpoint: {deepencoder_path}")

# Final stats
print0(f"Total training time: {total_training_time / 60:.2f}m")
print0(f"Minimum val loss: {min_val_loss:.4f}")

if min_val_loss < 0.1:
    print0("SUCCESS: Training converged!")

wandb_run.finish()
compute_cleanup()
