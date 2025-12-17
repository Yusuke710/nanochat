# Blueprint: nano-deepseek-ocr.py

A single-file, training-ready DeepSeek-OCR implementation combining DeepEncoder vision components with nanochat's GPT decoder. No vLLM dependencies.

---

## Architecture Overview

```
Image (H×W×3)
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DeepEncoder                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ SAM-ViT-B    │───▶│ CLIP-L/14    │───▶│ MLP Projector│──────▶│ Vision Tokens
│  │ (86M + 6M)   │    │ (~300M)      │    │ (2048→n_embd)│       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                  ▲                                     │
│         └──────────────────┘ (SAM provides patch features)       │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Embedding Merge                               │
│  [vision_tokens] merged at <image> token positions               │
│  + image_newline (learnable) + view_separator (learnable)        │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    nanochat GPT                                  │
│  (from nanochat/gpt.py - rotary, GQA, relu², no bias)           │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
Logits → Cross-Entropy Loss (same as nanochat)
```

---

## Required nanochat GPT Modification

**CRITICAL**: Before using this blueprint, nanochat's `gpt.py` must be modified to accept `inputs_embeds` parameter. This allows vision-language models to inject pre-merged vision+text embeddings.

See **[training_plan.md Section 5.0](training_plan.md#50-modify-nanochat-gpt-to-accept-inputs_embeds-critical)** for the complete modification.

**Summary of change** (in `gpt.py`):
```python
# BEFORE: Only accepts token IDs
def forward(self, idx, targets=None, ...):
    x = self.transformer.wte(idx)
    ...

# AFTER: Also accepts pre-computed embeddings
def forward(self, idx=None, inputs_embeds=None, targets=None, ...):
    if inputs_embeds is not None:
        x = inputs_embeds
    else:
        x = self.transformer.wte(idx)
    ...
```

This change is:
- **Backward compatible**: Original nanochat code works unchanged
- **Minimal**: ~10 lines modified
- **Standard**: Same pattern used by HuggingFace transformers

---

## File Structure

```
nanochat/                       (forked form nanochat, repo name is nano-deepseek-ocr)
├── nano_deepseek_ocr.py        # Main model file (this blueprint)
├── deepencoder/
│   ├── __init__.py
│   ├── sam_vary_sdpa.py        # SAM-ViT-B encoder (reuse from DeepSeek-OCR)
│   ├── clip_sdpa.py            # CLIP-L/14 encoder (reuse from DeepSeek-OCR)
│   ├── build_linear.py         # MLP projector (reuse from DeepSeek-OCR)
│   └── load_pretrained.py      # HuggingFace weight loading (from DeepEncoder_loading_plan.md)
├── image_process.py            # DeepseekOCRProcessor (adapted, remove hardcoded PROMPT)
├── config.py                   # Configuration constants
└── gpt.py                  # nanochat GPT (MODIFIED: add inputs_embeds support)
```

---

## nano_deepseek_ocr.py

```python
"""
nano-deepseek-ocr: Single-file vision-language model for OCR.

Combines:
- DeepEncoder: SAM-ViT-B + CLIP-L/14 + MLP Projector
- nanochat GPT: Decoder-only transformer

Training-ready, no vLLM dependencies.
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepencoder.sam_vary_sdpa import build_sam_vit_b
from deepencoder.clip_sdpa import build_clip_l
from deepencoder.build_linear import MlpProjector
from nanochat.gpt import GPT, GPTConfig
from addict import Dict


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DeepEncoderConfig:
    """DeepEncoder configuration."""
    # SAM config
    sam_embed_dim: int = 768
    sam_depth: int = 12
    sam_num_heads: int = 12

    # CLIP config (ViT-L/14)
    clip_embed_dim: int = 1024
    clip_depth: int = 24
    clip_num_heads: int = 16

    # Projector config
    projector_type: str = "linear"
    projector_input_dim: int = 2048  # CLIP (1024) + SAM compressed (1024)

    # Output dim matches nanochat GPT
    n_embd: int = 768  # Must match GPTConfig.n_embd


@dataclass
class NanoDeepseekOCRConfig:
    """Combined config for the full model."""
    # Vision encoder
    encoder: DeepEncoderConfig = None

    # Language model (nanochat GPT)
    gpt: GPTConfig = None

    # Special tokens
    image_token: str = "<image>"
    tile_tag: str = "2D"

    def __post_init__(self):
        if self.encoder is None:
            self.encoder = DeepEncoderConfig()
        if self.gpt is None:
            self.gpt = GPTConfig()
        # Ensure n_embd matches
        self.encoder.n_embd = self.gpt.n_embd


# =============================================================================
# DeepEncoder: Vision Encoder
# =============================================================================

class DeepEncoder(nn.Module):
    """
    Vision encoder combining SAM and CLIP.

    Flow:
    1. SAM extracts high-res local features
    2. SAM features are compressed and fed to CLIP (bypassing CLIP's patch_embed)
    3. CLIP + SAM features concatenated and projected to LLM dimension
    """

    def __init__(self, config: DeepEncoderConfig):
        super().__init__()
        self.config = config

        # SAM-ViT-B: local feature extractor (~92M params)
        # Includes net_2 and net_3 for compression (randomly initialized)
        self.sam = build_sam_vit_b()

        # CLIP-ViT-L/14: global feature extractor (~300M params)
        # patch_embedding is bypassed; takes SAM's compressed features
        self.clip = build_clip_l()

        # MLP Projector: projects concatenated features to LLM dim
        projector_cfg = Dict(
            projector_type=config.projector_type,
            input_dim=config.projector_input_dim,
            n_embed=config.n_embd,
        )
        self.projector = MlpProjector(projector_cfg)

        # Special embeddings for token sequence formatting
        embed_std = 1 / math.sqrt(config.n_embd)
        self.image_newline = nn.Parameter(torch.randn(config.n_embd) * embed_std)
        self.view_separator = nn.Parameter(torch.randn(config.n_embd) * embed_std)

    def encode_image(
        self,
        pixel_values: torch.Tensor,      # (B, 3, H, W) global view
        images_crop: torch.Tensor,        # (B, num_crops, 3, H, W) local crops
        images_spatial_crop: torch.Tensor # (B, 2) [width_tiles, height_tiles]
    ) -> torch.Tensor:
        """
        Encode images to vision tokens.

        Returns:
            vision_embeds: (B, num_vision_tokens, n_embd)
        """
        batch_size = pixel_values.size(0)
        device = pixel_values.device
        dtype = pixel_values.dtype

        vision_tokens_list = []

        for b in range(batch_size):
            # Get single sample
            global_img = pixel_values[b:b+1]  # (1, 3, H, W)
            crops = images_crop[b]             # (num_crops, 3, h, w)
            crop_shape = images_spatial_crop[b]  # [w_tiles, h_tiles]

            # Check if we have local crops
            has_crops = crops.sum().item() != 0 and (crop_shape[0] > 1 or crop_shape[1] > 1)

            if has_crops:
                # Process local crops through SAM → CLIP → Projector
                # SAM output: (num_crops, 1024, H', W') - channels first after conv compressor
                local_sam_feat = self.sam(crops)
                local_clip_feat = self.clip(crops, local_sam_feat)  # (num_crops, seq_len+1, 1024)

                # Concatenate CLIP (excluding CLS) with flattened SAM features
                # SAM: flatten(2) -> (B, 1024, H'*W'), permute -> (B, H'*W', 1024)
                local_combined = torch.cat([
                    local_clip_feat[:, 1:],  # Remove CLS token: (B, seq_len, 1024)
                    local_sam_feat.flatten(2).permute(0, 2, 1)  # (B, H'*W', 1024)
                ], dim=-1)  # -> (B, seq_len, 2048)
                local_features = self.projector(local_combined)  # (num_crops, seq_len, n_embd)

            # Process global view
            # SAM output: (1, 1024, H', W') - channels first
            global_sam_feat = self.sam(global_img)
            global_clip_feat = self.clip(global_img, global_sam_feat)
            global_combined = torch.cat([
                global_clip_feat[:, 1:],  # (1, seq_len, 1024)
                global_sam_feat.flatten(2).permute(0, 2, 1)  # (1, H'*W', 1024)
            ], dim=-1)  # -> (1, seq_len, 2048)
            global_features = self.projector(global_combined)  # (1, seq_len, n_embd)

            # Format tokens with newlines (2D tile format)
            tokens = self._format_vision_tokens(
                global_features.squeeze(0),
                local_features if has_crops else None,
                crop_shape if has_crops else None
            )
            vision_tokens_list.append(tokens)

        # Pad to same length and stack
        max_len = max(t.size(0) for t in vision_tokens_list)
        padded = torch.zeros(batch_size, max_len, self.config.n_embd, device=device, dtype=dtype)
        for b, tokens in enumerate(vision_tokens_list):
            padded[b, :tokens.size(0)] = tokens

        return padded

    def _format_vision_tokens(
        self,
        global_feat: torch.Tensor,   # (hw, n_embd)
        local_feat: Optional[torch.Tensor],  # (num_crops, hw, n_embd)
        crop_shape: Optional[torch.Tensor]   # [w_tiles, h_tiles]
    ) -> torch.Tensor:
        """Format vision tokens with newlines for 2D layout."""
        n_embd = global_feat.size(-1)
        hw = global_feat.size(0)
        h = w = int(math.sqrt(hw))

        # Reshape global features to 2D and add newline per row
        global_2d = global_feat.view(h, w, n_embd)
        global_with_newline = torch.cat([
            global_2d,
            self.image_newline.expand(h, 1, n_embd)
        ], dim=1)  # (h, w+1, n_embd)
        global_flat = global_with_newline.reshape(-1, n_embd)  # (h*(w+1), n_embd)

        if local_feat is None:
            # No crops: global + view_separator
            return torch.cat([global_flat, self.view_separator.unsqueeze(0)], dim=0)

        # Process local crops similarly
        w_tiles, h_tiles = int(crop_shape[0]), int(crop_shape[1])
        num_crops, hw2, _ = local_feat.shape
        h2 = w2 = int(math.sqrt(hw2))

        # Rearrange crops to spatial layout
        local_2d = local_feat.view(h_tiles, w_tiles, h2, w2, n_embd)
        local_2d = local_2d.permute(0, 2, 1, 3, 4)  # (h_tiles, h2, w_tiles, w2, n_embd)
        local_2d = local_2d.reshape(h_tiles * h2, w_tiles * w2, n_embd)

        # Add newlines
        local_with_newline = torch.cat([
            local_2d,
            self.image_newline.expand(h_tiles * h2, 1, n_embd)
        ], dim=1)
        local_flat = local_with_newline.reshape(-1, n_embd)

        # Combine: local + global + view_separator
        return torch.cat([
            local_flat,
            global_flat,
            self.view_separator.unsqueeze(0)
        ], dim=0)

    def freeze_sam(self):
        """Freeze SAM encoder (for Stage 2 training)."""
        for param in self.sam.parameters():
            param.requires_grad = False

    def unfreeze_sam(self):
        """Unfreeze SAM encoder."""
        for param in self.sam.parameters():
            param.requires_grad = True


# =============================================================================
# NanoDeepseekOCR: Full Vision-Language Model
# =============================================================================

class NanoDeepseekOCR(nn.Module):
    """
    Vision-Language model for OCR tasks.

    Combines DeepEncoder (vision) with nanochat GPT (language).
    Compatible with nanochat's training pipeline.
    """

    def __init__(self, config: NanoDeepseekOCRConfig):
        super().__init__()
        self.config = config

        # Vision encoder
        self.vision_encoder = DeepEncoder(config.encoder)

        # Language model (nanochat GPT)
        self.gpt = GPT(config.gpt)

        # Get image token ID (will be set by tokenizer)
        self.image_token_id: Optional[int] = None

    def set_image_token_id(self, token_id: int):
        """Set the image token ID from tokenizer."""
        self.image_token_id = token_id

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get text embeddings from GPT's embedding layer."""
        return self.gpt.transformer.wte(input_ids)

    def merge_vision_embeddings(
        self,
        input_ids: torch.Tensor,       # (B, T)
        text_embeds: torch.Tensor,     # (B, T, n_embd)
        vision_embeds: torch.Tensor,   # (B, num_vision_tokens, n_embd)
    ) -> torch.Tensor:
        """
        Merge vision embeddings into text embeddings at <image> token positions.

        The <image> tokens in input_ids are replaced with vision_embeds.
        """
        B, T, D = text_embeds.shape

        # Find image token positions
        image_mask = (input_ids == self.image_token_id)

        # Clone to avoid in-place modification
        merged = text_embeds.clone()

        for b in range(B):
            img_positions = image_mask[b].nonzero(as_tuple=True)[0]
            num_img_tokens = img_positions.size(0)

            if num_img_tokens > 0:
                # Replace image token embeddings with vision embeddings
                merged[b, img_positions] = vision_embeds[b, :num_img_tokens]

        return merged

    def forward(
        self,
        input_ids: torch.Tensor,                    # (B, T) token IDs
        targets: Optional[torch.Tensor] = None,     # (B, T) target token IDs
        pixel_values: Optional[torch.Tensor] = None,       # (B, 3, H, W)
        images_crop: Optional[torch.Tensor] = None,        # (B, num_crops, 3, h, w)
        images_spatial_crop: Optional[torch.Tensor] = None, # (B, 2)
        loss_reduction: str = 'mean',
    ):
        """
        Forward pass.

        Args:
            input_ids: Token IDs with <image> placeholder tokens
            targets: Target token IDs for loss computation
            pixel_values: Global view images
            images_crop: Local crop images
            images_spatial_crop: Crop grid dimensions [w_tiles, h_tiles]
            loss_reduction: 'mean' or 'none'

        Returns:
            If targets provided: loss (scalar or per-token)
            Otherwise: logits (B, T, vocab_size)
        """
        B, T = input_ids.shape

        # Get text embeddings
        text_embeds = self.get_input_embeddings(input_ids)

        # If images provided, encode and merge
        if pixel_values is not None:
            assert self.image_token_id is not None, "image_token_id not set"

            vision_embeds = self.vision_encoder.encode_image(
                pixel_values=pixel_values,
                images_crop=images_crop,
                images_spatial_crop=images_spatial_crop,
            )

            inputs_embeds = self.merge_vision_embeddings(
                input_ids, text_embeds, vision_embeds
            )
        else:
            inputs_embeds = text_embeds

        # Forward through GPT using pre-computed embeddings
        # NOTE: Requires modified gpt.py with inputs_embeds support (see training_plan.md Section 5.0)
        return self.gpt(inputs_embeds=inputs_embeds, targets=targets, loss_reduction=loss_reduction)

    def init_weights(self):
        """Initialize weights."""
        self.gpt.init_weights()
        # Vision encoder uses pretrained weights (loaded separately)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        images_crop: Optional[torch.Tensor] = None,
        images_spatial_crop: Optional[torch.Tensor] = None,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ):
        """
        Autoregressive generation.

        Yields tokens one at a time.
        """
        device = input_ids.device

        # Encode images once
        if pixel_values is not None:
            vision_embeds = self.vision_encoder.encode_image(
                pixel_values=pixel_values,
                images_crop=images_crop,
                images_spatial_crop=images_spatial_crop,
            )
        else:
            vision_embeds = None

        # Build initial embeddings with vision tokens
        text_embeds = self.get_input_embeddings(input_ids)
        if vision_embeds is not None:
            inputs_embeds = self.merge_vision_embeddings(input_ids, text_embeds, vision_embeds)
        else:
            inputs_embeds = text_embeds

        # Generation loop
        current_embeds = inputs_embeds

        for _ in range(max_tokens):
            # Forward through GPT with pre-computed embeddings
            logits = self.gpt(inputs_embeds=current_embeds)
            logits = logits[:, -1, :]  # Last position

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            yield next_token.item()

            # Append new token embedding
            next_embed = self.gpt.transformer.wte(next_token)
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)

    def setup_optimizers(
        self,
        matrix_lr: float = 0.02,
        embedding_lr: float = 0.2,
        unembedding_lr: float = 0.004,
        vision_lr: float = 0.001,
        weight_decay: float = 0.0,
    ):
        """
        Setup optimizers following nanochat's pattern.

        Uses Muon for matrix params, AdamW for embeddings.
        DDP-compatible: uses DistMuon/DistAdamW when running distributed.
        """
        from functools import partial
        from nanochat.common import get_dist_info
        from nanochat.muon import Muon, DistMuon
        from nanochat.adamw import DistAdamW

        ddp, rank, local_rank, world_size = get_dist_info()

        # Collect parameters by type
        vision_params = list(self.vision_encoder.parameters())
        gpt_matrix_params = list(self.gpt.transformer.h.parameters())
        gpt_embedding_params = list(self.gpt.transformer.wte.parameters())
        gpt_lm_head_params = list(self.gpt.lm_head.parameters())

        # Filter to only trainable
        vision_params = [p for p in vision_params if p.requires_grad]
        gpt_matrix_params = [p for p in gpt_matrix_params if p.requires_grad]
        gpt_embedding_params = [p for p in gpt_embedding_params if p.requires_grad]
        gpt_lm_head_params = [p for p in gpt_lm_head_params if p.requires_grad]

        # Scale LR for AdamW params by ∝1/√dmodel (nanochat pattern)
        model_dim = self.config.gpt.n_embd
        dmodel_lr_scale = (model_dim / 768) ** -0.5

        optimizers = []

        # Vision encoder uses AdamW (not Muon - vision encoders use standard optimizers)
        if vision_params:
            AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
            vision_opt = AdamWFactory(
                [{'params': vision_params, 'lr': vision_lr * dmodel_lr_scale}],
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=weight_decay,
            )
            optimizers.append(vision_opt)

        # GPT matrix params use Muon
        if gpt_matrix_params:
            MuonFactory = DistMuon if ddp else Muon
            muon_opt = MuonFactory(gpt_matrix_params, lr=matrix_lr, momentum=0.95)
            optimizers.append(muon_opt)

        # GPT embeddings + lm_head use AdamW
        if gpt_embedding_params or gpt_lm_head_params:
            adam_groups = []
            if gpt_lm_head_params:
                adam_groups.append({'params': gpt_lm_head_params, 'lr': unembedding_lr * dmodel_lr_scale})
            if gpt_embedding_params:
                adam_groups.append({'params': gpt_embedding_params, 'lr': embedding_lr * dmodel_lr_scale})

            AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
            adam_opt = AdamWFactory(
                adam_groups,
                betas=(0.8, 0.95),
                eps=1e-10,
                weight_decay=weight_decay,
            )
            optimizers.append(adam_opt)

        # Store initial_lr for LR scheduling (nanochat pattern)
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]

        return optimizers


# =============================================================================
# Factory Functions
# =============================================================================

def build_nano_deepseek_ocr(
    gpt_checkpoint: Optional[str] = None,
    load_vision_pretrained: bool = True,
    n_embd: int = 768,
    n_layer: int = 12,
    n_head: int = 6,
    n_kv_head: int = 6,
    vocab_size: int = 50304,
    sequence_len: int = 8192,
) -> NanoDeepseekOCR:
    """
    Build NanoDeepseekOCR model.

    Args:
        gpt_checkpoint: Path to nanochat GPT checkpoint
        load_vision_pretrained: Load SAM/CLIP from HuggingFace
        n_embd: GPT embedding dimension
        n_layer: Number of GPT layers
        n_head: Number of attention heads
        n_kv_head: Number of KV heads (for GQA)
        vocab_size: Vocabulary size (extended for vision tokens)
        sequence_len: Maximum sequence length

    Returns:
        NanoDeepseekOCR model
    """
    # Build configs
    encoder_config = DeepEncoderConfig(n_embd=n_embd)
    gpt_config = GPTConfig(
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        vocab_size=vocab_size,
        sequence_len=sequence_len,
    )
    config = NanoDeepseekOCRConfig(encoder=encoder_config, gpt=gpt_config)

    # Build model
    model = NanoDeepseekOCR(config)

    # Load pretrained vision weights
    if load_vision_pretrained:
        from deepencoder.load_pretrained import (
            load_sam_weights_from_hf,
            load_clip_weights_from_hf
        )
        load_sam_weights_from_hf(model.vision_encoder.sam)
        load_clip_weights_from_hf(model.vision_encoder.clip)
        print("Loaded pretrained SAM and CLIP weights from HuggingFace")

    # Load GPT checkpoint
    if gpt_checkpoint:
        gpt_state = torch.load(gpt_checkpoint)
        model.gpt.load_state_dict(gpt_state, strict=False)
        print(f"Loaded GPT weights from {gpt_checkpoint}")

    return model


def load_from_huggingface(
    gpt_model_id: str = "nanochat-students/base-d20",
    device: str = "cuda",
) -> NanoDeepseekOCR:
    """
    Load model with GPT from HuggingFace.

    Extends tokenizer with vision tokens automatically.
    """
    from transformers import AutoTokenizer

    # Load tokenizer and extend with vision tokens
    tokenizer = AutoTokenizer.from_pretrained(gpt_model_id)

    # Add vision special tokens
    vision_tokens = [
        "<image>",
        "<|grounding|>",
        "<|ref|>", "<|/ref|>",
        "<|det|>", "<|/det|>",
    ]
    tokenizer.add_special_tokens({"additional_special_tokens": vision_tokens})

    # Build model with extended vocab
    model = build_nano_deepseek_ocr(
        gpt_checkpoint=None,  # Will load from HF
        load_vision_pretrained=True,
        vocab_size=len(tokenizer),
    )

    # Set image token ID
    model.set_image_token_id(tokenizer.convert_tokens_to_ids("<image>"))

    # TODO: Load GPT weights from HuggingFace
    # This requires adapting the checkpoint format

    model.to(device)
    return model, tokenizer


# =============================================================================
# Training Stage Setup (in vis_tok_train.py / vis_mid_train.py)
# =============================================================================

def setup_stage1(model: NanoDeepseekOCR):
    """
    Stage 1: Train ALL components (DeepEncoder + GPT).
    Called in vis_tok_train.py.

    Trains: SAM, CLIP, Projector, special embeddings, GPT (all parameters)
    Frozen: Nothing
    """
    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True


def setup_stage2(model: NanoDeepseekOCR):
    """
    Stage 2: Fine-tune CLIP + Projector + GPT, freeze SAM.
    Called in vis_mid_train.py.

    Trains: CLIP, Projector, special embeddings, GPT
    Frozen: SAM
    """
    # Freeze SAM
    model.vision_encoder.freeze_sam()

    # Unfreeze CLIP and projector
    for param in model.vision_encoder.clip.parameters():
        param.requires_grad = True
    for param in model.vision_encoder.projector.parameters():
        param.requires_grad = True
    model.vision_encoder.image_newline.requires_grad = True
    model.vision_encoder.view_separator.requires_grad = True

    # Unfreeze GPT
    for param in model.gpt.parameters():
        param.requires_grad = True


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Build model
    model = build_nano_deepseek_ocr(
        load_vision_pretrained=True,
        n_embd=768,
        n_layer=12,
    )

    # Setup for Stage 1 training (in vis_tok_train.py)
    setup_stage1(model)
    optimizers = model.setup_optimizers()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
```

---

## Vocabulary Extension

### Required Special Tokens

Vision-language models require extending the tokenizer vocabulary with special tokens:

```python
# Vision special tokens (must be added to tokenizer)
VISION_SPECIAL_TOKENS = [
    "<image>",          # Image placeholder (expands to N tokens based on image size)
    "<|grounding|>",    # For grounding tasks
    "<|ref|>", "<|/ref|>",  # Reference markers
    "<|det|>", "<|/det|>",  # Detection markers
]

# These exist in nanochat tokenizer already:
# <|bos|>, <|eos|>, <|user_start|>, <|user_end|>, <|assistant_start|>, <|assistant_end|>
```

### Extending the Tokenizer

```python
from transformers import AutoTokenizer

def extend_tokenizer_for_vision(tokenizer_path: str) -> AutoTokenizer:
    """
    Extend nanochat tokenizer with vision special tokens.

    Returns tokenizer with new vocab_size for model initialization.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Add vision tokens
    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": VISION_SPECIAL_TOKENS
    })
    print(f"Added {num_added} vision tokens. New vocab size: {len(tokenizer)}")

    return tokenizer
```

### Model Embedding Resize

When vocab_size increases, the embedding and lm_head layers must be resized. This is the **recommended approach** because it preserves pretrained nanochat weights:

```python
def resize_embeddings(model: NanoDeepseekOCR, new_vocab_size: int):
    """
    Resize model embeddings for extended vocabulary.

    Called AFTER loading pretrained GPT weights but BEFORE training.
    Preserves pretrained embeddings and initializes new tokens properly.
    """
    old_vocab_size = model.gpt.config.vocab_size

    if new_vocab_size == old_vocab_size:
        return

    # Resize token embedding
    old_wte = model.gpt.transformer.wte
    new_wte = nn.Embedding(new_vocab_size, model.gpt.config.n_embd)
    new_wte.weight.data[:old_vocab_size] = old_wte.weight.data
    # Initialize new tokens with small random values
    nn.init.normal_(new_wte.weight.data[old_vocab_size:], mean=0.0, std=0.02)
    model.gpt.transformer.wte = new_wte

    # Resize lm_head
    old_lm_head = model.gpt.lm_head
    new_lm_head = nn.Linear(model.gpt.config.n_embd, new_vocab_size, bias=False)
    new_lm_head.weight.data[:old_vocab_size] = old_lm_head.weight.data
    nn.init.zeros_(new_lm_head.weight.data[old_vocab_size:])  # Zero-init like nanochat
    model.gpt.lm_head = new_lm_head

    # Update config
    model.gpt.config.vocab_size = new_vocab_size
    model.config.gpt.vocab_size = new_vocab_size

    print(f"Resized embeddings: {old_vocab_size} -> {new_vocab_size}")


# Usage: Load pretrained GPT, then extend vocabulary
model = build_nano_deepseek_ocr(
    gpt_checkpoint="path/to/nanochat.pt",
    vocab_size=50304,  # Original nanochat vocab
    ...
)
tokenizer = extend_tokenizer_for_vision("nanochat-students/base-d20")
resize_embeddings(model, len(tokenizer))
model.set_image_token_id(tokenizer.convert_tokens_to_ids("<image>"))
```

This approach:
1. **Preserves pretrained weights**: nanochat embeddings remain intact
2. **Follows VLM convention**: LLaVA, Qwen-VL use the same resize pattern
3. **Clean initialization**: New tokens get proper random init (embedding) and zero init (lm_head)

---

## Key Design Decisions

### 1. Single-File Simplicity
Following nanochat's `gpt.py` philosophy: all core logic in one file, minimal abstractions.

### 2. Training-First Design
- `setup_stage1()` in `vis_tok_train.py` / `setup_stage2()` in `vis_mid_train.py` for training stage control
- `model.setup_optimizers()` returns Muon + AdamW like nanochat
- Forward pass returns loss directly when targets provided

### 3. Minimal Changes to DeepEncoder
- Reuse `sam_vary_sdpa.py`, `clip_sdpa.py`, `build_linear.py` as-is
- Only wrap them in a clean `DeepEncoder` class

### 4. Compatible with nanochat Training Loop
```python
# Training loop (same pattern as nanochat)
for batch in dataloader:
    input_ids, targets, pixel_values, images_crop, images_spatial_crop = batch

    loss = model(
        input_ids=input_ids,
        targets=targets,
        pixel_values=pixel_values,
        images_crop=images_crop,
        images_spatial_crop=images_spatial_crop,
    )

    loss.backward()
    for opt in optimizers:
        opt.step()
        opt.zero_grad()
```

### 5. No vLLM Dependencies
Removed all:
- `vllm.config`
- `vllm.model_executor`
- `vllm.multimodal`
- `MULTIMODAL_REGISTRY`
- `AutoWeightsLoader`

---

## Changes to image_process.py

The original `image_process.py` has hardcoded `PROMPT` in `tokenize_with_images`.
Modify to accept prompt as parameter:

```python
def tokenize_with_images(
    self,
    images: List[Image.Image],
    prompt: str,  # NEW: Accept prompt as parameter
    bos: bool = True,
    eos: bool = True,
    cropping: bool = True,
):
    """Tokenize text with <image> tags."""
    conversation = prompt  # Use passed prompt instead of global PROMPT
    # ... rest unchanged
```

---

## Integration with nanochat Training

See `training_plan.md` for detailed training pipeline. Key points:
- Use nanochat's `TaskMixture` for data mixing
- Vision tasks return `{prompt, response, image}` format
- Text tasks return `{messages}` conversation format
- Collate function handles mixed formats

---

## Next Steps

1. **Implement `deepencoder/load_pretrained.py`** per `DeepEncoder_loading_plan.md`
2. **Modify `image_process.py`** to accept prompt parameter
3. **Create training script** following `training_plan.md`
4. **Test on small data** to verify end-to-end flow
