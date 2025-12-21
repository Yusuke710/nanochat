# vLLM Inference for nano-deepseek-ocr (Sidequest)

This documents the feasibility of using DeepSeek-OCR's vLLM implementation with nano-deepseek-ocr checkpoints.

## Reference Code Location

`nanochat/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/`

## ⚠️ Architecture Incompatibility

**The existing vLLM DeepSeek-OCR code CANNOT work with nano-deepseek-ocr weights** due to fundamental architecture differences:

| Component | DeepSeek-OCR (vLLM) | nano-deepseek-ocr |
|-----------|---------------------|-------------------|
| Language Model | DeepSeek V2/V3 | nanochat GPT |
| Attention | Multi-head Latent Attention (MLA) | Standard Q/K/V |
| Architecture | MoE (Mixture of Experts) | Dense MLP |
| Layer naming | `language.model.layers.{i}.*` | `gpt.transformer.h.{i}.*` |
| Config | DeepseekVLV2Config | GPTConfig |

The vLLM code (`deepseek_ocr.py:314-326`) explicitly initializes:
```python
architectures = ["DeepseekV3ForCausalLM"]  # or V2
self.language_model = init_vllm_registered_model(...)
```

Simply renaming weight keys will NOT work because the layer structures are incompatible.

## Vision Encoder Compatibility

The vision encoder components ARE compatible:
- `sam_model.*` - SAM ViT-B encoder ✓
- `vision_model.*` - CLIP ViT-L encoder ✓
- `projector.*` - MLP projector ✓
- `image_newline` / `view_separator` - special tokens ✓

## Current Inference (Working)

Use the native nanochat inference with KV-cache Engine:

```bash
# Test on vlm-overfit10 dataset
python -m scripts.vision_sample --step 1500

# Or use Engine directly
from nanochat.engine import Engine
engine = Engine(model, tokenizer)
for token, _ in engine.generate(prompt_ids, pixel_values=pv, max_tokens=512):
    ...
```

## Options for vLLM Support

### Option 1: Register nanochat GPT with vLLM (Recommended)

Create a new vLLM model class for nanochat GPT:
1. Implement `NanochatGPTForCausalLM` following vLLM patterns
2. Register with `ModelRegistry.register_model()`
3. Modify `DeepseekOCRForCausalLM` to use nanochat GPT

### Option 2: Convert to HuggingFace Format

Export nano-deepseek-ocr to HuggingFace-compatible format:
1. Create `config.json` with model architecture
2. Save weights as `model.safetensors`
3. Use standard HuggingFace/vLLM loading

### Option 3: Use Native Inference (Current)

The `nanochat.engine.Engine` class already provides:
- KV-cache for efficient autoregressive generation
- Streaming token output
- Temperature/top-k sampling

## Benchmarks

| Method | Tokens/sec | Notes |
|--------|------------|-------|
| Naive (no cache) | ~50 | Full forward pass each step |
| Engine (KV cache) | ~200 | Cached attention |
| vLLM | N/A | Not compatible |

## TODO

- [ ] Implement Option 1 or 2 for vLLM support
- [ ] Benchmark against native Engine inference
