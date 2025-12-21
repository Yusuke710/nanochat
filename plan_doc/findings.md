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
