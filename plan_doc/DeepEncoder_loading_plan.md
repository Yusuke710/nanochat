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

HuggingFace uses separate q/k/v projections, DeepSeek uses fused `qkv_proj`:

```python
# Fuse q, k, v into single qkv_proj
q_w = hf_state[f"encoder.layers.{i}.self_attn.q_proj.weight"]
k_w = hf_state[f"encoder.layers.{i}.self_attn.k_proj.weight"]
v_w = hf_state[f"encoder.layers.{i}.self_attn.v_proj.weight"]
new_state[f"transformer.layers.{i}.self_attn.qkv_proj.weight"] = torch.cat([q_w, k_w, v_w], dim=0)
```

## Weight Name Mappings

### SAM

```
patch_embed.projection.weight  → patch_embed.proj.weight
layers.{i}.layer_norm1         → blocks.{i}.norm1
layers.{i}.layer_norm2         → blocks.{i}.norm2
neck.conv1.weight              → neck.0.weight
neck.layer_norm1               → neck.1
```

### CLIP

```
embeddings.class_embedding           → embeddings.class_embedding
embeddings.position_embedding.weight → embeddings.position_embedding.weight
encoder.layers.{i}.*                 → transformer.layers.{i}.*
self_attn.q/k/v_proj                 → self_attn.qkv_proj (fused)
```

## Usage

```python
from deepencoder.load_pretrained import load_sam_weights_from_hf, load_clip_weights_from_hf

sam = build_sam_vit_b()
clip = build_clip_l()

load_sam_weights_from_hf(sam)   # net_2, net_3 stay random
load_clip_weights_from_hf(clip)  # QKV fused automatically
```
