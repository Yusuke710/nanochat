# DeepEncoder Weight Loading Plan

Load original SAM ViT-B (~92M) and CLIP ViT-L/14 (~300M) weights from HuggingFace into DeepSeek-OCR's custom architecture for training.

## HuggingFace Models

| Model | HuggingFace ID | Params | Architecture |
|-------|----------------|--------|--------------|
| SAM ViT-B | `facebook/sam-vit-base` | ~92M (vision encoder + extra layers) | 768 embed, 12 layers, 12 heads |
| CLIP ViT-L/14 | `openai/clip-vit-large-patch14` | ~300M (vision encoder, patch_embed bypassed) | 1024 embed, 24 layers, 16 heads |

## Compatibility Analysis

### SAM: Partial Match (~92M total)

DeepSeek's SAM (`sam_vary_sdpa.py`) has **extra layers** not in original:
- `net_2`: Conv2d(256→512, stride=2) - ~1.2M params - **random init**
- `net_3`: Conv2d(512→1024, stride=2) - ~4.7M params - **random init**

**From HuggingFace:** `patch_embed`, `pos_embed`, `blocks.*`, `neck.*` (~86M)
**Random init (for training):** `net_2`, `net_3` (~6M)

### CLIP: Architecture Match (~300M)

DeepSeek's CLIP (`clip_sdpa.py`) matches ViT-L/14 but:
- `patch_embedding` layer is **bypassed** at runtime (SAM provides patch features)
- Different layer naming convention

## Weight Name Mapping

### SAM: HuggingFace → DeepSeek

```
# Patch embedding
patch_embed.projection.weight        → patch_embed.proj.weight
patch_embed.projection.bias          → patch_embed.proj.bias

# Position embedding
pos_embed                            → pos_embed

# Transformer blocks (i = 0..11)
layers.{i}.layer_norm1.weight        → blocks.{i}.norm1.weight
layers.{i}.layer_norm1.bias          → blocks.{i}.norm1.bias
layers.{i}.attn.qkv.weight           → blocks.{i}.attn.qkv.weight
layers.{i}.attn.qkv.bias             → blocks.{i}.attn.qkv.bias
layers.{i}.attn.proj.weight          → blocks.{i}.attn.proj.weight
layers.{i}.attn.proj.bias            → blocks.{i}.attn.proj.bias
layers.{i}.layer_norm2.weight        → blocks.{i}.norm2.weight
layers.{i}.layer_norm2.bias          → blocks.{i}.norm2.bias
layers.{i}.mlp.lin1.weight           → blocks.{i}.mlp.lin1.weight
layers.{i}.mlp.lin1.bias             → blocks.{i}.mlp.lin1.bias
layers.{i}.mlp.lin2.weight           → blocks.{i}.mlp.lin2.weight
layers.{i}.mlp.lin2.bias             → blocks.{i}.mlp.lin2.bias

# Relative position (per block with global attention: 2, 5, 8, 11)
layers.{i}.attn.rel_pos_h            → blocks.{i}.attn.rel_pos_h
layers.{i}.attn.rel_pos_w            → blocks.{i}.attn.rel_pos_w

# Neck
neck.conv1.weight                    → neck.0.weight
neck.layer_norm1.weight              → neck.1.weight
neck.layer_norm1.bias                → neck.1.bias
neck.conv2.weight                    → neck.2.weight
neck.layer_norm2.weight              → neck.3.weight
neck.layer_norm2.bias                → neck.3.bias

# NOT in HuggingFace (random init)
# net_2.weight
# net_3.weight
```

### CLIP: HuggingFace → DeepSeek

```
# Embeddings
embeddings.class_embedding                          → embeddings.class_embedding
embeddings.patch_embedding.weight                   → embeddings.patch_embedding.weight  # bypassed at runtime
embeddings.position_embedding.weight                → embeddings.position_embedding.weight

# Pre-layernorm
pre_layrnorm.weight                                 → pre_layrnorm.weight
pre_layrnorm.bias                                   → pre_layrnorm.bias

# Transformer blocks (i = 0..23)
encoder.layers.{i}.layer_norm1.weight               → transformer.layers.{i}.layer_norm1.weight
encoder.layers.{i}.layer_norm1.bias                 → transformer.layers.{i}.layer_norm1.bias
encoder.layers.{i}.self_attn.q_proj.weight          → transformer.layers.{i}.self_attn.qkv_proj.weight  # needs concat
encoder.layers.{i}.self_attn.k_proj.weight          → (combined into qkv_proj)
encoder.layers.{i}.self_attn.v_proj.weight          → (combined into qkv_proj)
encoder.layers.{i}.self_attn.q_proj.bias            → transformer.layers.{i}.self_attn.qkv_proj.bias   # needs concat
encoder.layers.{i}.self_attn.k_proj.bias            → (combined into qkv_proj)
encoder.layers.{i}.self_attn.v_proj.bias            → (combined into qkv_proj)
encoder.layers.{i}.self_attn.out_proj.weight        → transformer.layers.{i}.self_attn.out_proj.weight
encoder.layers.{i}.self_attn.out_proj.bias          → transformer.layers.{i}.self_attn.out_proj.bias
encoder.layers.{i}.layer_norm2.weight               → transformer.layers.{i}.layer_norm2.weight
encoder.layers.{i}.layer_norm2.bias                 → transformer.layers.{i}.layer_norm2.bias
encoder.layers.{i}.mlp.fc1.weight                   → transformer.layers.{i}.mlp.fc1.weight
encoder.layers.{i}.mlp.fc1.bias                     → transformer.layers.{i}.mlp.fc1.bias
encoder.layers.{i}.mlp.fc2.weight                   → transformer.layers.{i}.mlp.fc2.weight
encoder.layers.{i}.mlp.fc2.bias                     → transformer.layers.{i}.mlp.fc2.bias
```

**Note:** HuggingFace CLIP uses separate q/k/v projections, DeepSeek uses fused `qkv_proj`. Need to concatenate weights.

## Implementation Code

### File: `deepencoder/load_pretrained.py`

```python
"""Load pretrained SAM and CLIP weights from HuggingFace."""
import torch
from transformers import SamModel, CLIPModel


def load_sam_weights_from_hf(sam_model, hf_model_id="facebook/sam-vit-base"):
    """
    Load SAM ViT-B weights from HuggingFace into DeepSeek's SAM.

    Args:
        sam_model: DeepSeek's ImageEncoderViT instance
        hf_model_id: HuggingFace model identifier

    Returns:
        sam_model with loaded weights (net_2, net_3 remain random)
    """
    print(f"Loading SAM weights from {hf_model_id}...")
    hf_sam = SamModel.from_pretrained(hf_model_id)
    hf_state = hf_sam.vision_encoder.state_dict()

    # Build mapping
    new_state = {}

    for hf_name, tensor in hf_state.items():
        ds_name = _map_sam_name(hf_name)
        if ds_name is not None:
            new_state[ds_name] = tensor

    # Load with strict=False (net_2, net_3 stay random)
    missing, unexpected = sam_model.load_state_dict(new_state, strict=False)

    print(f"SAM: Loaded {len(new_state)} tensors")
    print(f"SAM: Missing (random init): {missing}")

    return sam_model


def _map_sam_name(hf_name):
    """Map HuggingFace SAM weight name to DeepSeek name."""

    # Patch embedding
    if hf_name == "patch_embed.projection.weight":
        return "patch_embed.proj.weight"
    if hf_name == "patch_embed.projection.bias":
        return "patch_embed.proj.bias"

    # Position embedding
    if hf_name == "pos_embed":
        return "pos_embed"

    # Transformer blocks
    if hf_name.startswith("layers."):
        # layers.{i}.layer_norm1 -> blocks.{i}.norm1
        new_name = hf_name.replace("layers.", "blocks.")
        new_name = new_name.replace(".layer_norm1.", ".norm1.")
        new_name = new_name.replace(".layer_norm2.", ".norm2.")
        return new_name

    # Neck
    if hf_name.startswith("neck."):
        # neck.conv1 -> neck.0, neck.layer_norm1 -> neck.1, etc.
        mapping = {
            "neck.conv1.weight": "neck.0.weight",
            "neck.layer_norm1.weight": "neck.1.weight",
            "neck.layer_norm1.bias": "neck.1.bias",
            "neck.conv2.weight": "neck.2.weight",
            "neck.layer_norm2.weight": "neck.3.weight",
            "neck.layer_norm2.bias": "neck.3.bias",
        }
        return mapping.get(hf_name)

    return None


def load_clip_weights_from_hf(clip_model, hf_model_id="openai/clip-vit-large-patch14"):
    """
    Load CLIP ViT-L/14 weights from HuggingFace into DeepSeek's CLIP.

    Args:
        clip_model: DeepSeek's VitModel instance
        hf_model_id: HuggingFace model identifier

    Returns:
        clip_model with loaded weights
    """
    print(f"Loading CLIP weights from {hf_model_id}...")
    hf_clip = CLIPModel.from_pretrained(hf_model_id)
    hf_state = hf_clip.vision_model.state_dict()

    # Build mapping
    new_state = {}

    # Handle QKV fusion per layer
    num_layers = 24

    for i in range(num_layers):
        # Fuse q, k, v projections into qkv_proj
        q_w = hf_state[f"encoder.layers.{i}.self_attn.q_proj.weight"]
        k_w = hf_state[f"encoder.layers.{i}.self_attn.k_proj.weight"]
        v_w = hf_state[f"encoder.layers.{i}.self_attn.v_proj.weight"]
        new_state[f"transformer.layers.{i}.self_attn.qkv_proj.weight"] = torch.cat([q_w, k_w, v_w], dim=0)

        q_b = hf_state[f"encoder.layers.{i}.self_attn.q_proj.bias"]
        k_b = hf_state[f"encoder.layers.{i}.self_attn.k_proj.bias"]
        v_b = hf_state[f"encoder.layers.{i}.self_attn.v_proj.bias"]
        new_state[f"transformer.layers.{i}.self_attn.qkv_proj.bias"] = torch.cat([q_b, k_b, v_b], dim=0)

    # Map remaining weights
    for hf_name, tensor in hf_state.items():
        ds_name = _map_clip_name(hf_name)
        if ds_name is not None and ds_name not in new_state:
            new_state[ds_name] = tensor

    # Load weights
    missing, unexpected = clip_model.load_state_dict(new_state, strict=False)

    print(f"CLIP: Loaded {len(new_state)} tensors")
    if missing:
        print(f"CLIP: Missing: {missing}")

    return clip_model


def _map_clip_name(hf_name):
    """Map HuggingFace CLIP weight name to DeepSeek name."""

    # Skip q/k/v projections (handled separately for fusion)
    if "self_attn.q_proj" in hf_name or "self_attn.k_proj" in hf_name or "self_attn.v_proj" in hf_name:
        return None

    # Embeddings
    if hf_name == "embeddings.class_embedding":
        return "embeddings.class_embedding"
    if hf_name == "embeddings.patch_embedding.weight":
        return "embeddings.patch_embedding.weight"
    if hf_name == "embeddings.position_embedding.weight":
        return "embeddings.position_embedding.weight"

    # Pre-layernorm
    if hf_name.startswith("pre_layrnorm."):
        return hf_name  # Same name

    # Transformer blocks
    if hf_name.startswith("encoder.layers."):
        new_name = hf_name.replace("encoder.layers.", "transformer.layers.")
        new_name = new_name.replace(".self_attn.out_proj.", ".self_attn.out_proj.")
        return new_name

    return None
```

### Modify: `deepencoder/sam_vary_sdpa.py`

At line 481, change:

```python
def build_sam_vit_b(checkpoint=None, load_hf_pretrained=False):
    """Build SAM ViT-B model.

    Args:
        checkpoint: Path to local checkpoint file
        load_hf_pretrained: If True, load weights from HuggingFace
    """
    model = _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=None,  # Don't load in _build_sam
    )

    if load_hf_pretrained:
        from .load_pretrained import load_sam_weights_from_hf
        model = load_sam_weights_from_hf(model)
    elif checkpoint is not None:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(
            {k[30:]: v for k, v in state_dict.items() if 'vision_tower_high' in k},
            strict=True
        )
        print(f"Loaded SAM from {checkpoint}")

    return model
```

### Modify: `deepencoder/clip_sdpa.py`

At line 447, change:

```python
def build_clip_l(checkpoint=None, load_hf_pretrained=False):
    """Build CLIP ViT-L/14 model.

    Args:
        checkpoint: Path to local checkpoint file
        load_hf_pretrained: If True, load weights from HuggingFace
    """
    model = VitModel(
        cfg=vit_model_cfg,
        freeze_embed=False,
        freeze_pre_norm=False,
    )

    if load_hf_pretrained:
        from .load_pretrained import load_clip_weights_from_hf
        model = load_clip_weights_from_hf(model)
    elif checkpoint is not None:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)
        print(f"Loaded CLIP from {checkpoint}")

    return model
```

## Usage

```python
# Load models with HuggingFace pretrained weights
sam_model = build_sam_vit_b(load_hf_pretrained=True)
clip_model = build_clip_l(load_hf_pretrained=True)

# Or in DeepseekOCRForCausalLM.__init__
self.sam_model = build_sam_vit_b(load_hf_pretrained=True)
self.vision_model = build_clip_l(load_hf_pretrained=True)
```

## Verification

```python
# Verify SAM loading (base weights should match, net_2/net_3 random)
from transformers import SamModel
import torch

hf_sam = SamModel.from_pretrained("facebook/sam-vit-base")
ds_sam = build_sam_vit_b(load_hf_pretrained=True)

x = torch.randn(1, 3, 1024, 1024)
with torch.no_grad():
    hf_out = hf_sam.vision_encoder(x).last_hidden_state  # Before neck
    # ds_sam output will differ due to net_2/net_3

# Verify CLIP loading
from transformers import CLIPModel

hf_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
ds_clip = build_clip_l(load_hf_pretrained=True)

x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    hf_out = hf_clip.vision_model(x).last_hidden_state
    ds_out = ds_clip(x, None)  # patch_embeds=None uses internal patch_embedding
    print(f"CLIP outputs match: {torch.allclose(hf_out, ds_out, atol=1e-5)}")
```

## Notes

1. **SAM `net_2` and `net_3`**: Randomly initialized, will be trained
2. **CLIP `patch_embedding`**: Loaded but bypassed at runtime when SAM provides features
3. **QKV fusion**: HuggingFace CLIP uses separate q/k/v, DeepSeek uses fused qkv - weights are concatenated
4. **Vary-base is not open-sourced**: The original DeepSeek-OCR loads SAM weights from Vary's `vision_tower_high` component, but Vary-base model weights are not publicly available. We use Meta's original SAM ViT-B weights (`facebook/sam-vit-base`) instead, which share the same architecture. The `net_2` and `net_3` compression layers (added by Vary/DeepSeek on top of SAM) will be trained from scratch.
