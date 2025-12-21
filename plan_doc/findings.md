# Findings

## FineVision Dataset Testing (2025-12-21)

### Hardware
- GPU: NVIDIA A100-SXM4-80GB (80GB VRAM)
- Storage: 100GB (constraint that required using only chartqa dataset)

### Dataset Configuration
Due to storage constraints, olmOCR-mix-0225-documents was removed from training:
- **Stage 1**: chartqa only (18K samples for train, 100 for val)
- **Stage 2**: chartqa + SmolTalk (mixed vision + text training)

### Training Results

| Stage | Script | Steps | Time | Final Loss | Min Val Loss | Best Step |
|-------|--------|-------|------|------------|--------------|-----------|
| 1 | vis_tok_train.py | 500 | 3.02m | ~1.90 | 1.8278 | 200 |
| 2 | vis_mid_train.py | 1000 | 4.73m | ~2.22 | 1.9604 | 700 |

### Stage 1: Vision Token Training (vis_tok_train.py)
- All components trained: SAM encoder, CLIP encoder, projector, GPT decoder
- Learning rate: 5e-5 with cosine annealing
- Batch size: 10, seq_len: 4096
- Outputs: `deepencoder_500.pt` (encoder-only checkpoint for Stage 2)

### Stage 2: Vision Mid Training (vis_mid_train.py)
- SAM encoder frozen, only CLIP, projector, and fresh GPT trained
- Fresh nanochat GPT loaded from HuggingFace (Stage 1 decoder discarded per DeepSeek-OCR paper)
- Learning rate: 3e-5 with StepLR decay
- Batch size: 4, seq_len: 8192 (longer sequences)
- Mixed vision + text batches via masked_scatter

### Checkpoints
```
checkpoints/
├── step_100.pt - step_500.pt (Stage 1 full model)
├── deepencoder_500.pt (Stage 1 encoder-only, used by Stage 2)
├── step_500.pt (Stage 2 intermediate)
└── step_1000.pt (Stage 2 final)
```

### Observations
- Both stages show good convergence with validation loss under 2.0
- Training is stable on A100 80GB with bfloat16 autocast
- FineVision HuggingFace dataset wrapper works correctly with start/stop slicing
- SmolTalk text-only batches integrate seamlessly with vision batches
