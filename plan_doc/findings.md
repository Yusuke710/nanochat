# Findings

## FineVision Dataset Testing (2025-12-21)

### Hardware
- GPU: NVIDIA A100-SXM4-80GB (80GB VRAM)
- Storage: 100GB (constraint that required using only chartqa dataset)

### Dataset Configuration
Due to storage constraints, olmOCR-mix-0225-documents was removed from training:
- **Stage 1**: chartqa only (18K samples for train, 100 for val)
- **Stage 2**: chartqa + SmolTalk (mixed vision + text training)

### Training Results (with WandB logging)

| Stage | Script | Steps | Batch | Time | Final Loss | Min Val Loss | Best Step |
|-------|--------|-------|-------|------|------------|--------------|-----------|
| 1 | vis_tok_train.py | 1000 | 10 | 13.36m | ~1.45 | 1.4569 | 850 |
| 2 | vis_mid_train.py | 1500 | 2 | ~5m | ~2.15 | 2.1596 | 1400 |

### WandB Integration
- Project: `nano-deepseek-ocr`
- Stage 1 run: https://wandb.ai/youfu1202mo-unsw/nano-deepseek-ocr/runs/errja0lq
- Stage 2 run: https://wandb.ai/youfu1202mo-unsw/nano-deepseek-ocr/runs/avd4hc6b
- Metrics logged: train/loss, train/lr, train/dt, train/tok_per_sec, train/mfu, train/total_flops, val/loss

### Stage 1: Vision Token Training (vis_tok_train.py)
- All components trained: SAM encoder, CLIP encoder, projector, GPT decoder
- Learning rate: 5e-5 with cosine annealing
- Batch size: 10, seq_len: 4096
- Outputs: `deepencoder_1000.pt` (encoder-only checkpoint for Stage 2)

### Stage 2: Vision Mid Training (vis_mid_train.py)
- SAM encoder frozen, only CLIP, projector, and fresh GPT trained
- Fresh nanochat GPT loaded from HuggingFace (Stage 1 decoder discarded per DeepSeek-OCR paper)
- Learning rate: 3e-5 with StepLR decay
- Batch size: 2 (reduced from 4 due to OOM with seq_len=8192)
- seq_len: 8192 (longer sequences)
- Mixed vision + text batches via masked_scatter

### Memory Considerations
- Stage 2 with batch_size=4 and seq_len=8192 causes OOM on A100 80GB
- Reduced to batch_size=2 for stable training

### Checkpoints
```
checkpoints/
├── step_*.pt (Stage 1 full model checkpoints)
├── deepencoder_1000.pt (Stage 1 encoder-only, used by Stage 2)
└── step_1500.pt (Stage 2 final)
```

### Observations
- Both stages show good convergence with validation loss under 2.2
- Training is stable on A100 80GB with bfloat16 autocast
- FineVision HuggingFace dataset wrapper works correctly with start/stop slicing
- SmolTalk text-only batches integrate seamlessly with vision batches

---

## Evaluation Pipeline Testing (2025-12-21)

### Hardware
- GPU: NVIDIA H100 NVL (95GB VRAM)

### Training Configuration
- Stage 1: 1000 steps on ChartQA
- Stage 2: 2000 steps on ChartQA + SmolTalk

### Benchmark Evaluation Results

| Benchmark | Accuracy (1-NED) | NED | F1 | Dataset |
|-----------|------------------|-----|-----|---------|
| **Fox** | 0.08% | 0.9992 | 0.0017 | ucaslcl/Fox_benchmark_data |
| **OmniDocBench** | 1.06% | 0.9894 | 0.0236 | Quivr/OmniDocBench (981 samples) |

Note: Model outputs `'0.6.'` for all inputs - essentially non-functional.

### Critical Issues Found

#### 1. Precision Metric is Misleading (Fixed)
**Problem**: Original code used character-level precision as primary metric for Fox.
- Model outputs: `'0.6.'` (4 chars)
- Ground truth: 5000+ chars
- Precision = 95%+ (because `.`, `0`, `6` exist in most text)
- But accuracy is actually 0%

**Fix**: Updated `scripts/vision_eval.py` to:
- Use accuracy = 1-NED as primary metric
- Show sample outputs for debugging
- Mark precision as "misleading if output is short"

#### 2. OmniDocBench Dataset Source (Fixed)
**Problem**: `opendatalab/OmniDocBench` HF dataset only contains images (no annotations).

**Fix**: Switched to `Quivr/OmniDocBench` which has:
- Annotations in Parquet format (`layout_dets` field with text)
- Images in `images/` folder (downloaded via `hf_hub_download`)
- 981 samples with full layout annotations
- Much faster evaluation (4.4s vs 285s for 5 samples)

### Bug Fixes Applied
1. `vis_tok_train.py:211` - DataLoader iterator fix
2. `vis_mid_train.py:263` - DataLoader iterator fix
3. `vis_mid_train.py:267-268` - Device transfer for val_inputs/val_targets
4. `vision_eval.py:113` - torch.autocast device_type string fix
5. `scripts/vision_eval.py` - Use meaningful metrics (1-NED) instead of precision
6. `tasks/omnidocbench.py` - Use Quivr/OmniDocBench (has annotations in Parquet)

### Conclusion
Model trained on limited data (ChartQA only) for limited steps (3000 total) is essentially non-functional for general OCR. This is expected - the model just repeats `'0.6.'` regardless of input. Proper training would require:
- Full dataset (olmOCR-mix, DocVQA, etc.)
- Many more training steps (10K-100K)
- Proper hyperparameter tuning

---

## Data Loading Pattern Analysis (2025-12-21)

### Karpathy's nanochat Pattern vs PyTorch DataLoader

Two different patterns for moving tensors to device:

#### 1. Custom Generator (Karpathy's approach in mid_train.py, chat_sft.py)
```python
# Inside generator - data returned already on GPU
inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=True)
targets = targets_cpu.view(B, T).to(device=device, non_blocking=True)
yield inputs, targets

# Training loop - no .to(device) needed
x, y = next(train_loader)
loss = model(x, y)
```

#### 2. PyTorch DataLoader with Workers (multimodal_dataloader.py)
```python
# Collate function returns CPU tensors (workers can't access GPU)
return inputs, targets, media  # CPU tensors

# Training loop - must move to device
for inputs, targets, media in train_loader:
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
```

### Why the Difference?
- **Custom generators**: Run in main process, can access GPU directly
- **PyTorch DataLoader workers**: Separate processes, cannot access GPU
- `pin_memory=True` + `non_blocking=True` enables async CPU→GPU transfer

### Bug Fixed
`vis_tok_train.py` validation loop was missing `.to(device)` calls:
```python
# Before (bug)
val_inputs, val_targets, val_media = next(val_loader)
# val_inputs/val_targets still on CPU!

# After (fixed)
val_inputs = val_inputs.to(device, non_blocking=True)
val_targets = val_targets.to(device, non_blocking=True)
```

### Decision
Use PyTorch DataLoader pattern (move to device in training loop) for vision training scripts since:
- More scalable with parallel data loading
- Standard PyTorch practice
- Works with `num_workers > 0`
