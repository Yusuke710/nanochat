"""
Vision Mid Training (Stage 2) - Train with SAM frozen.

Fresh start: Load DeepEncoder from Stage 1 + fresh nanochat GPT from HuggingFace.
Resume: Load full checkpoint directly (skip HF download).

Usage:
    # Fresh start (requires deepencoder checkpoint)
    python -m scripts.vis_mid_train --resume_from_deepencoder=checkpoints/deepencoder_300.pt

    # Resume from full checkpoint
    python -m scripts.vis_mid_train --resume_step=500

Distributed:
    torchrun --nproc_per_node=8 -m scripts.vis_mid_train --resume_from_deepencoder=...

Stage 2 differences from Stage 1:
    - SAM encoder frozen (only CLIP, projector, GPT trained)
    - Fresh GPT loaded from HuggingFace (Stage 1 decoder discarded)
    - StepLR scheduler instead of constant LR
    - Lower learning rate (3e-5)
    - Longer sequences (8192)
    - Mixed vision + text training (configurable ratio)
"""

import os
import time
from contextlib import nullcontext

import torch

from nanochat.gpt import GPTConfig
from nanochat.nano_deepseek_ocr import build_nano_deepseek_ocr
from nanochat.deepencoder.load_pretrained import load_nanochat_gpt_from_hf
from nanochat.multimodal_dataloader import create_multimodal_loader
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.loss_eval import evaluate_bpb_multimodal
from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type, DummyWandb
import wandb
from tasks.finevision import FineVision
from tasks.smoltalk import SmolTalk
from tasks.common import TaskMixture

# -----------------------------------------------------------------------------
# User settings (overridable via CLI)
run = "dummy"  # wandb run name ("dummy" = no wandb logging)
# Model
base_size = 1024  # image resolution
seq_len = 8192  # longer sequences for stage 2
# Training
steps = 5000  # number of training steps
batch_size = 8  # batch size (smaller due to longer sequences)
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
resume_from_deepencoder = ""  # DeepEncoder checkpoint path (REQUIRED for Stage 2)
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

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nano-deepseek-ocr", name=run, config=user_config)

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

# -----------------------------------------------------------------------------
# Load model weights
start_step = 0

if resume_step >= 0:
    # Resume from full checkpoint (skip deepencoder + HF loading)
    # First expand embeddings to match checkpoint shape
    if vocab_size > pretrained_vocab_size:
        print0(f"Expanding embeddings: {pretrained_vocab_size} -> {vocab_size}")
        model.gpt.transformer.wte = torch.nn.Embedding(vocab_size, gpt_config.n_embd)
        model.gpt.lm_head = torch.nn.Linear(gpt_config.n_embd, vocab_size, bias=False)
    # Load full checkpoint
    ckpt_path = os.path.join(checkpoint_dir, f"step_{resume_step}.pt")
    print0(f"Resuming from full checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    start_step = resume_step
else:
    # Fresh start: load deepencoder + fresh GPT from HuggingFace
    assert resume_from_deepencoder, "Fresh start requires --resume_from_deepencoder=<path>"

    # Load trained DeepEncoder (SAM, CLIP, projector, special tokens)
    print0(f"Loading DeepEncoder: {resume_from_deepencoder}")
    deepencoder_state = torch.load(resume_from_deepencoder, map_location="cpu", weights_only=False)
    model.load_state_dict(deepencoder_state, strict=False)

    # Load fresh nanochat GPT from HuggingFace
    print0("Loading fresh nanochat GPT from HuggingFace...")
    hf_token = os.getenv("HF_TOKEN")
    model.gpt = load_nanochat_gpt_from_hf(model.gpt, hf_token=hf_token, verbose=True)

    # Expand embeddings for new tokens
    if vocab_size > pretrained_vocab_size:
        print0(f"Expanding embeddings: {pretrained_vocab_size} -> {vocab_size}")
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

# Load token bytes for BPB computation
token_bytes = get_token_bytes(device=device)

# -----------------------------------------------------------------------------
# Wrap with DDP for multi-GPU training
if ddp:
    # Stage 2 mixes vision + text, so vision encoder may not get gradients on text-only steps
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[ddp_local_rank],
        find_unused_parameters=True
    )
raw_model = model.module if ddp else model  # unwrapped model for saving/attributes

# Model stats
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
tokens_per_batch = batch_size * seq_len * ddp_world_size
# Use GPT's estimate_flops for transformer part (Chinchilla formula)
num_flops_per_token = raw_model.gpt.estimate_flops()
print0(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")
print0(f"Tokens per batch: {tokens_per_batch:,}")
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# H100 theoretical peak (bfloat16, no sparsity)
promised_flops_per_sec = 989e12 * ddp_world_size

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
# Task Mixture (define your tasks here - task-agnostic design)
# Modify this section to change what data is used for training
# Mixed batches (vision + text) are supported via masked_scatter
# Note: FineVision uses start/stop (only has train split), SmolTalk uses split="test" for val
train_ds = TaskMixture([
    FineVision("chartqa", stop=18000),  # chartqa train (~18K samples)
    SmolTalk(split="train", stop=10000),  # first 10K text samples
])
val_ds = TaskMixture([
    FineVision("chartqa", start=18000, stop=18100),  # chartqa val (100 samples, no overlap)
    SmolTalk(split="test", stop=100),  # 100 samples from test split (no overlap)
])

# -----------------------------------------------------------------------------
# Setup dataloaders (unified multimodal pipeline with PyTorch DataLoader)
train_task_names = [t.__class__.__name__ for t in train_ds.tasks]
val_task_names = [t.__class__.__name__ for t in val_ds.tasks]
print0(f"Train tasks: {train_task_names}, Val tasks: {val_task_names}")
print0(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

train_loader = create_multimodal_loader(train_ds, tokenizer, batch_size, seq_len, base_size)
val_loader_fn = lambda: create_multimodal_loader(val_ds, tokenizer, batch_size, seq_len, base_size)

# Verify first batch
inputs, targets, media = next(iter(train_loader))
pixel_values = media["pixel_values"]
print0(f"Batch shapes: inputs={inputs.shape}, targets={targets.shape}, pixel_values={pixel_values.shape if pixel_values is not None else None}")

# -----------------------------------------------------------------------------
# Loop state
min_val_bpb = float("inf")
smooth_train_loss = 0.0
total_training_time = 0.0

# -----------------------------------------------------------------------------
# Training loop
print0(f"\nStarting Stage 2 training for {steps} steps...")
model.train()
train_iter = iter(train_loader)

for step in range(start_step, steps):
    last_step = (step == steps - 1)

    # -------------------------------------------------------------------------
    # Evaluation
    if eval_every > 0 and (last_step or (step > 0 and step % eval_every == 0)):
        model.eval()
        with autocast_ctx:
            val_bpb = evaluate_bpb_multimodal(model, val_loader_fn(), eval_steps, token_bytes, device)
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        wandb_run.log({"step": step, "val/bpb": val_bpb})
        model.train()

    # -------------------------------------------------------------------------
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
    final_path = os.path.join(checkpoint_dir, f"step_{steps}.pt")
    torch.save(raw_model.state_dict(), final_path)
    print0(f"\nFinal checkpoint saved to {final_path}")

# Final stats
print0(f"Total training time: {total_training_time / 60:.2f}m")
print0(f"Minimum val bpb: {min_val_bpb:.4f}")

if min_val_bpb < 1.0:
    print0("SUCCESS: Stage 2 training converged!")

wandb_run.finish()
compute_cleanup()
