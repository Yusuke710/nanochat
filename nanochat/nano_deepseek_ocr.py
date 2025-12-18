# Copied from https://github.com/deepseek-ai/DeepSeek-OCR
# Modifications:
# - Simplified architecture for nanochat integration
# - Uses nanochat GPT instead of DeepSeek LLM
# - Removed VLLM-specific code
# - Added training-focused forward pass with loss computation

"""
NanoDeepseekOCR - Vision Language Model combining SAM + CLIP vision encoder with nanochat GPT.

Architecture:
    Image (H x W x 3)
        |
        v
    SAM-ViT-B (768 embed, 12 layers) -> (H/64, W/64, 1024) after net_2, net_3 compression
        |
        v
    CLIP-L/14 (1024 hidden, 24 layers) -> (num_patches + 1, 1024)
        |
        v
    Concatenate SAM + CLIP features -> (num_patches, 2048)
        |
        v
    Linear Projector -> (num_patches, n_embd)
        |
        v
    Replace <image> token positions in GPT input embeddings
        |
        v
    nanochat GPT (~570M params)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

from nanochat.gpt import GPT, GPTConfig
from nanochat.deepencoder.sam_vary_sdpa import build_sam_vit_b
from nanochat.deepencoder.clip_sdpa import build_clip_l
from nanochat.deepencoder.build_linear import MlpProjector


@dataclass
class VisionConfig:
    """Configuration for the vision encoder."""
    base_size: int = 1024  # Image size after padding
    patch_size: int = 16   # Patch size for vision encoder
    downsample_ratio: int = 4  # SAM compression ratio
    sam_embed_dim: int = 1024  # SAM output dimension (after net_3)
    clip_embed_dim: int = 1024  # CLIP hidden dimension


class NanoDeepseekOCR(nn.Module):
    """
    Vision Language Model combining:
    - SAM ViT-B vision encoder with compression layers
    - CLIP ViT-L/14 for semantic features
    - Linear projector to GPT embedding dimension
    - nanochat GPT language model
    """

    def __init__(
        self,
        gpt_config: GPTConfig,
        vision_config: Optional[VisionConfig] = None,
        image_token_id: int = None,
    ):
        super().__init__()

        self.gpt_config = gpt_config
        self.vision_config = vision_config or VisionConfig()
        self.image_token_id = image_token_id

        # Vision encoders
        self.sam_model = build_sam_vit_b()
        self.vision_model = build_clip_l()

        # Projector: concatenated SAM + CLIP features (2048) -> GPT embedding dim
        projector_cfg = edict(
            projector_type="linear",
            input_dim=self.vision_config.sam_embed_dim + self.vision_config.clip_embed_dim,
            n_embed=gpt_config.n_embd,
        )
        self.projector = MlpProjector(projector_cfg)

        # Special tokens for image formatting (2D tile tag style)
        embed_std = 1.0 / math.sqrt(gpt_config.n_embd)
        self.image_newline = nn.Parameter(torch.randn(gpt_config.n_embd) * embed_std)
        self.view_separator = nn.Parameter(torch.randn(gpt_config.n_embd) * embed_std)

        # Language model
        self.gpt = GPT(gpt_config)

    def set_image_token_id(self, token_id: int):
        """Set the image token ID after tokenizer is configured."""
        self.image_token_id = token_id

    def init_weights(self):
        """Initialize GPT weights. Vision encoder weights are loaded separately."""
        self.gpt.init_weights()

    def num_vision_tokens(self) -> int:
        """Calculate the number of vision tokens for a single image."""
        num_queries = (
            self.vision_config.base_size //
            self.vision_config.patch_size //
            self.vision_config.downsample_ratio
        )
        # (num_queries + 1) for each row (including newline) * num_queries rows + 1 separator
        return (num_queries + 1) * num_queries + 1

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode images through SAM + CLIP + projector.

        Args:
            pixel_values: (B, 3, H, W) normalized image tensors

        Returns:
            vision_embeds: (B, num_tokens, n_embd) vision embeddings
        """
        B = pixel_values.size(0)

        # SAM encoder (produces features at lower resolution)
        sam_features = self.sam_model(pixel_values)  # (B, 1024, H/64, W/64)

        # CLIP encoder (takes SAM features as patch embeddings)
        clip_features = self.vision_model(pixel_values, sam_features)  # (B, num_patches+1, 1024)

        # Remove CLS token from CLIP, keep patch features
        clip_patch_features = clip_features[:, 1:]  # (B, num_patches, 1024)

        # Flatten SAM features
        sam_flat = sam_features.flatten(2).permute(0, 2, 1)  # (B, num_patches, 1024)

        # Concatenate SAM and CLIP features
        combined = torch.cat([clip_patch_features, sam_flat], dim=-1)  # (B, num_patches, 2048)

        # Project to GPT dimension
        projected = self.projector(combined)  # (B, num_patches, n_embd)

        # Reshape and add newline tokens (2D tile format)
        _, hw, n_embd = projected.shape
        h = w = int(math.sqrt(hw))

        projected = projected.view(B, h, w, n_embd)

        # Add newline token at the end of each row
        newline = self.image_newline.view(1, 1, 1, n_embd).expand(B, h, 1, n_embd)
        projected = torch.cat([projected, newline], dim=2)  # (B, h, w+1, n_embd)
        projected = projected.view(B, -1, n_embd)  # (B, h*(w+1), n_embd)

        # Add view separator at the end
        separator = self.view_separator.view(1, 1, n_embd).expand(B, 1, n_embd)
        vision_embeds = torch.cat([projected, separator], dim=1)  # (B, h*(w+1)+1, n_embd)

        return vision_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        vision_embeds: Optional[torch.Tensor] = None,
        loss_reduction: str = 'mean',
    ):
        """
        Forward pass for training and inference.

        Args:
            input_ids: (B, T) token IDs with <image> tokens expanded
            targets: (B, T) target token IDs for loss computation
            pixel_values: (B, 3, H, W) normalized image tensors
            vision_embeds: (B, num_vision_tokens, n_embd) pre-computed vision embeddings (for efficient generation)
            loss_reduction: 'mean', 'sum', or 'none'

        Returns:
            If targets provided: loss
            Otherwise: logits
        """
        B, T = input_ids.size()

        # Get text embeddings from GPT
        text_embeds = self.gpt.transformer.wte(input_ids)  # (B, T, n_embd)

        # If images provided, encode and merge with text embeddings
        if pixel_values is not None or vision_embeds is not None:
            assert self.image_token_id is not None, "Image token ID not set"

            # Use pre-computed vision embeddings if available, otherwise encode
            if vision_embeds is None:
                vision_embeds = self.encode_images(pixel_values)  # (B, num_vision_tokens, n_embd)

            # Find image token positions and replace with vision embeddings
            image_mask = (input_ids == self.image_token_id)

            # Flatten vision embeddings across batch for scatter
            # We assume one image per sample for now
            flat_vision = vision_embeds.reshape(-1, vision_embeds.size(-1))

            # Create output tensor and scatter vision embeddings
            text_embeds = text_embeds.clone()
            text_embeds[image_mask] = flat_vision.to(text_embeds.dtype)

        # Forward through GPT with inputs_embeds
        # We need to modify the GPT forward to accept inputs_embeds
        return self._forward_gpt_with_embeds(text_embeds, targets, loss_reduction)

    def _forward_gpt_with_embeds(
        self,
        inputs_embeds: torch.Tensor,
        targets: Optional[torch.Tensor],
        loss_reduction: str = 'mean',
    ):
        """
        Forward pass through GPT starting from embeddings instead of token IDs.

        This mirrors the GPT.forward() logic but skips the embedding lookup.
        """
        B, T, _ = inputs_embeds.size()
        config = self.gpt.config

        # Grab rotary embeddings
        assert T <= self.gpt.cos.size(1), f"Sequence too long: {T} > {self.gpt.cos.size(1)}"
        cos_sin = self.gpt.cos[:, :T], self.gpt.sin[:, :T]

        # Forward through transformer
        x = inputs_embeds
        x = F.rms_norm(x, (x.size(-1),))  # norm after embedding
        for block in self.gpt.transformer.h:
            x = block(x, cos_sin, kv_cache=None)
        x = F.rms_norm(x, (x.size(-1),))  # final norm

        # Compute logits
        softcap = 15
        logits = self.gpt.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction
            )
            return loss
        else:
            return logits

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: (1, T) input token IDs
            pixel_values: (1, 3, H, W) image tensor
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling

        Returns:
            Generated token IDs including input
        """
        device = input_ids.device

        # Pre-compute vision embeddings once (major optimization!)
        vision_embeds = None
        if pixel_values is not None:
            vision_embeds = self.encode_images(pixel_values)

        for _ in range(max_new_tokens):
            # Forward pass using cached vision embeddings
            logits = self.forward(input_ids, vision_embeds=vision_embeds)
            logits = logits[:, -1, :]  # (1, vocab_size)

            # Apply temperature and top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_id], dim=1)

        return input_ids

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        gpt_config: Optional[GPTConfig] = None,
        vision_config: Optional[VisionConfig] = None,
        device: str = "cuda",
    ):
        """Load model from checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location=device)

        if gpt_config is None:
            # Try to infer from checkpoint
            gpt_config = GPTConfig()  # Use defaults

        model = cls(gpt_config, vision_config)
        model.load_state_dict(state_dict)
        return model


def build_nano_deepseek_ocr(
    gpt_config: Optional[GPTConfig] = None,
    vision_config: Optional[VisionConfig] = None,
) -> NanoDeepseekOCR:
    """
    Build NanoDeepseekOCR model with default configurations.

    Uses nanochat base-d20 config by default.
    """
    if gpt_config is None:
        # nanochat base-d20 config (no GQA, n_kv_head = n_head)
        gpt_config = GPTConfig(
            sequence_len=4096,
            vocab_size=65536,  # Base vocab size, will be checked after adding <image> token
            n_layer=20,
            n_head=16,
            n_kv_head=16,  # No GQA in pretrained model
            n_embd=1280,
        )

    model = NanoDeepseekOCR(gpt_config, vision_config)
    return model


if __name__ == "__main__":
    # Test model creation
    model = build_nano_deepseek_ocr()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test forward pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Create dummy inputs
    batch_size = 1
    seq_len = 300
    n_vision_tokens = model.num_vision_tokens()

    input_ids = torch.randint(0, 50304, (batch_size, seq_len), device=device)
    pixel_values = torch.randn(batch_size, 3, 1024, 1024, device=device)

    # Set a dummy image token ID
    image_token_id = 50303
    model.set_image_token_id(image_token_id)

    # Place image tokens
    input_ids[:, :n_vision_tokens] = image_token_id

    targets = input_ids.clone()
    targets[:, :n_vision_tokens] = -1  # Mask image tokens in targets

    print(f"Testing forward pass...")
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        loss = model(input_ids, targets=targets, pixel_values=pixel_values)
    print(f"Loss: {loss.item():.4f}")
