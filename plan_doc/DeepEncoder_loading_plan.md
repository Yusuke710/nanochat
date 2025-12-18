# DeepEncoder Weight Loading

Load SAM and CLIP from HuggingFace into DeepSeek-OCR's architecture.

## HuggingFace Models

| Model | HuggingFace ID | Params |
|-------|----------------|--------|
| SAM ViT-B | `facebook/sam-vit-base` | ~92M |
| CLIP ViT-L/14 | `openai/clip-vit-large-patch14` | ~300M |

## SAM: What's Loaded vs Random Init

**From HuggingFace**: `patch_embed`, `pos_embed`, `blocks.*`, `neck.*` (~86M)

**Random init**: `net_2`, `net_3` (~6M) - compression layers added by DeepSeek

## CLIP: QKV Fusion Required

HuggingFace uses separate q/k/v projections, DeepSeek uses fused `qkv_proj`.
The loading function fuses q, k, v weights into single qkv_proj.

## Weight Name Mappings

### SAM

```
HuggingFace                      → DeepSeek-OCR
─────────────────────────────────────────────────
patch_embed.projection.weight    → patch_embed.proj.weight
layers.{i}.layer_norm1           → blocks.{i}.norm1
layers.{i}.layer_norm2           → blocks.{i}.norm2
neck.conv1.weight                → neck.0.weight
neck.layer_norm1                 → neck.1
```

### CLIP

```
HuggingFace                           → DeepSeek-OCR
──────────────────────────────────────────────────────
embeddings.class_embedding            → embeddings.class_embedding
embeddings.position_embedding.weight  → embeddings.position_embedding.weight
encoder.layers.{i}.*                  → transformer.layers.{i}.*
self_attn.q/k/v_proj                  → self_attn.qkv_proj (fused)
```

## Implementation

See `nanochat/deepencoder/load_pretrained.py` for:
- `load_sam_weights_from_hf()` - net_2, net_3 stay random
- `load_clip_weights_from_hf()` - QKV fused automatically
- `load_nanochat_gpt_from_hf()` - loads GPT from nanochat-students/base-d20
