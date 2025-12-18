# Copied from https://github.com/deepseek-ai/DeepSeek-OCR
# Modifications:
# - Added functions to load SAM and CLIP weights from HuggingFace Hub
# - Added weight name mappings for SAM (facebook/sam-vit-base) and CLIP (openai/clip-vit-large-patch14)

"""
Load SAM and CLIP pretrained weights from HuggingFace.
"""

import os
import torch
from huggingface_hub import hf_hub_download


def load_sam_weights_from_hf(sam_model, hf_token=None, verbose=True):
    """
    Load SAM ViT-B weights from facebook/sam-vit-base into the DeepSeek-OCR SAM architecture.

    The net_2 and net_3 (compression layers added by DeepSeek) stay randomly initialized.

    Args:
        sam_model: The SAM model (ImageEncoderViT) to load weights into
        hf_token: HuggingFace token for authentication
        verbose: Print progress messages
    """
    from transformers import SamModel

    if verbose:
        print("Loading SAM ViT-B weights from facebook/sam-vit-base...")

    # Load HuggingFace SAM model
    hf_sam = SamModel.from_pretrained("facebook/sam-vit-base", token=hf_token)
    hf_state = hf_sam.vision_encoder.state_dict()

    # Build mapping from HuggingFace names to our model names
    new_state = {}

    # Patch embedding
    # HF: patch_embed.projection.weight -> ours: patch_embed.proj.weight
    new_state["patch_embed.proj.weight"] = hf_state["patch_embed.projection.weight"]
    new_state["patch_embed.proj.bias"] = hf_state["patch_embed.projection.bias"]

    # Position embedding
    new_state["pos_embed"] = hf_state["pos_embed"]

    # Transformer blocks
    for i in range(12):  # SAM ViT-B has 12 layers
        hf_prefix = f"layers.{i}"
        our_prefix = f"blocks.{i}"

        # Layer norms
        new_state[f"{our_prefix}.norm1.weight"] = hf_state[f"{hf_prefix}.layer_norm1.weight"]
        new_state[f"{our_prefix}.norm1.bias"] = hf_state[f"{hf_prefix}.layer_norm1.bias"]
        new_state[f"{our_prefix}.norm2.weight"] = hf_state[f"{hf_prefix}.layer_norm2.weight"]
        new_state[f"{our_prefix}.norm2.bias"] = hf_state[f"{hf_prefix}.layer_norm2.bias"]

        # Attention QKV (HF has separate, we have fused)
        q_w = hf_state[f"{hf_prefix}.attn.qkv.weight"]
        q_b = hf_state[f"{hf_prefix}.attn.qkv.bias"]
        new_state[f"{our_prefix}.attn.qkv.weight"] = q_w
        new_state[f"{our_prefix}.attn.qkv.bias"] = q_b

        # Attention projection
        new_state[f"{our_prefix}.attn.proj.weight"] = hf_state[f"{hf_prefix}.attn.proj.weight"]
        new_state[f"{our_prefix}.attn.proj.bias"] = hf_state[f"{hf_prefix}.attn.proj.bias"]

        # Relative position embeddings (if present)
        if f"{hf_prefix}.attn.rel_pos_h" in hf_state:
            new_state[f"{our_prefix}.attn.rel_pos_h"] = hf_state[f"{hf_prefix}.attn.rel_pos_h"]
            new_state[f"{our_prefix}.attn.rel_pos_w"] = hf_state[f"{hf_prefix}.attn.rel_pos_w"]

        # MLP
        new_state[f"{our_prefix}.mlp.lin1.weight"] = hf_state[f"{hf_prefix}.mlp.lin1.weight"]
        new_state[f"{our_prefix}.mlp.lin1.bias"] = hf_state[f"{hf_prefix}.mlp.lin1.bias"]
        new_state[f"{our_prefix}.mlp.lin2.weight"] = hf_state[f"{hf_prefix}.mlp.lin2.weight"]
        new_state[f"{our_prefix}.mlp.lin2.bias"] = hf_state[f"{hf_prefix}.mlp.lin2.bias"]

    # Neck (post-encoder conv layers)
    new_state["neck.0.weight"] = hf_state["neck.conv1.weight"]
    new_state["neck.1.weight"] = hf_state["neck.layer_norm1.weight"]
    new_state["neck.1.bias"] = hf_state["neck.layer_norm1.bias"]
    new_state["neck.2.weight"] = hf_state["neck.conv2.weight"]
    new_state["neck.3.weight"] = hf_state["neck.layer_norm2.weight"]
    new_state["neck.3.bias"] = hf_state["neck.layer_norm2.bias"]

    # Load the mapped weights (strict=False because net_2 and net_3 are not in HF model)
    missing, unexpected = sam_model.load_state_dict(new_state, strict=False)

    if verbose:
        print(f"SAM weights loaded. Missing keys (expected): {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")

    # Clean up
    del hf_sam, hf_state

    return sam_model


def load_clip_weights_from_hf(clip_model, hf_token=None, verbose=True):
    """
    Load CLIP ViT-L/14 weights from openai/clip-vit-large-patch14 into the DeepSeek-OCR CLIP architecture.

    Note: HuggingFace CLIP uses separate q/k/v projections, but our model uses fused qkv_proj.

    Args:
        clip_model: The CLIP model (VitModel) to load weights into
        hf_token: HuggingFace token for authentication
        verbose: Print progress messages
    """
    from transformers import CLIPModel

    if verbose:
        print("Loading CLIP ViT-L/14 weights from openai/clip-vit-large-patch14...")

    # Load HuggingFace CLIP model
    hf_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", token=hf_token)
    hf_state = hf_clip.vision_model.state_dict()

    # Build mapping from HuggingFace names to our model names
    new_state = {}

    # Embeddings
    new_state["embeddings.class_embedding"] = hf_state["embeddings.class_embedding"]
    new_state["embeddings.position_embedding.weight"] = hf_state["embeddings.position_embedding.weight"]
    new_state["embeddings.patch_embedding.weight"] = hf_state["embeddings.patch_embedding.weight"]

    # Pre-layernorm
    new_state["pre_layrnorm.weight"] = hf_state["pre_layrnorm.weight"]
    new_state["pre_layrnorm.bias"] = hf_state["pre_layrnorm.bias"]

    # Transformer layers (CLIP ViT-L has 24 layers)
    for i in range(24):
        hf_prefix = f"encoder.layers.{i}"
        our_prefix = f"transformer.layers.{i}"

        # Layer norms
        new_state[f"{our_prefix}.layer_norm1.weight"] = hf_state[f"{hf_prefix}.layer_norm1.weight"]
        new_state[f"{our_prefix}.layer_norm1.bias"] = hf_state[f"{hf_prefix}.layer_norm1.bias"]
        new_state[f"{our_prefix}.layer_norm2.weight"] = hf_state[f"{hf_prefix}.layer_norm2.weight"]
        new_state[f"{our_prefix}.layer_norm2.bias"] = hf_state[f"{hf_prefix}.layer_norm2.bias"]

        # Attention: fuse q, k, v into qkv_proj
        q_w = hf_state[f"{hf_prefix}.self_attn.q_proj.weight"]
        k_w = hf_state[f"{hf_prefix}.self_attn.k_proj.weight"]
        v_w = hf_state[f"{hf_prefix}.self_attn.v_proj.weight"]
        q_b = hf_state[f"{hf_prefix}.self_attn.q_proj.bias"]
        k_b = hf_state[f"{hf_prefix}.self_attn.k_proj.bias"]
        v_b = hf_state[f"{hf_prefix}.self_attn.v_proj.bias"]

        # Concatenate q, k, v weights and biases
        new_state[f"{our_prefix}.self_attn.qkv_proj.weight"] = torch.cat([q_w, k_w, v_w], dim=0)
        new_state[f"{our_prefix}.self_attn.qkv_proj.bias"] = torch.cat([q_b, k_b, v_b], dim=0)

        # Attention output projection
        new_state[f"{our_prefix}.self_attn.out_proj.weight"] = hf_state[f"{hf_prefix}.self_attn.out_proj.weight"]
        new_state[f"{our_prefix}.self_attn.out_proj.bias"] = hf_state[f"{hf_prefix}.self_attn.out_proj.bias"]

        # MLP
        new_state[f"{our_prefix}.mlp.fc1.weight"] = hf_state[f"{hf_prefix}.mlp.fc1.weight"]
        new_state[f"{our_prefix}.mlp.fc1.bias"] = hf_state[f"{hf_prefix}.mlp.fc1.bias"]
        new_state[f"{our_prefix}.mlp.fc2.weight"] = hf_state[f"{hf_prefix}.mlp.fc2.weight"]
        new_state[f"{our_prefix}.mlp.fc2.bias"] = hf_state[f"{hf_prefix}.mlp.fc2.bias"]

    # Load the mapped weights
    missing, unexpected = clip_model.load_state_dict(new_state, strict=False)

    if verbose:
        print(f"CLIP weights loaded. Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")

    # Clean up
    del hf_clip, hf_state

    return clip_model


def load_nanochat_gpt_from_hf(gpt_model, hf_token=None, verbose=True):
    """
    Load nanochat GPT weights from nanochat-students/base-d20.

    Args:
        gpt_model: The GPT model to load weights into
        hf_token: HuggingFace token for authentication
        verbose: Print progress messages
    """
    from huggingface_hub import hf_hub_download

    if verbose:
        print("Loading nanochat GPT weights from nanochat-students/base-d20...")

    # Download the model weights (using pytorch_model.bin)
    weights_path = hf_hub_download(
        repo_id="nanochat-students/base-d20",
        filename="pytorch_model.bin",
        token=hf_token
    )

    # Load the weights
    state_dict = torch.load(weights_path, map_location="cpu")

    # Load into model
    missing, unexpected = gpt_model.load_state_dict(state_dict, strict=False)

    if verbose:
        if missing:
            print(f"Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        print("GPT weights loaded successfully.")

    return gpt_model


if __name__ == "__main__":
    # Test the loading functions
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    print("Testing SAM weight loading...")
    from nanochat.deepencoder.sam_vary_sdpa import build_sam_vit_b
    sam = build_sam_vit_b()
    sam = load_sam_weights_from_hf(sam, hf_token=hf_token)
    print(f"SAM model has {sum(p.numel() for p in sam.parameters()):,} parameters")

    print("\nTesting CLIP weight loading...")
    from nanochat.deepencoder.clip_sdpa import build_clip_l
    clip = build_clip_l()
    clip = load_clip_weights_from_hf(clip, hf_token=hf_token)
    print(f"CLIP model has {sum(p.numel() for p in clip.parameters()):,} parameters")
