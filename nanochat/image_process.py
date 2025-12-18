# Copied from https://github.com/deepseek-ai/DeepSeek-OCR
# Modifications:
# - Simplified for nanochat integration
# - Removed VLLM-specific code
# - Added simple process_image function for inference

"""
Image processing utilities for nano-deepseek-ocr.
"""

import math
from typing import List, Tuple, Optional

import torch
import torchvision.transforms as T
from PIL import Image, ImageOps


class ImageTransform:
    """Transform PIL images to normalized tensors."""

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize: bool = True
    ):
        self.mean = mean
        self.std = std
        self.normalize = normalize

        transform_pipelines = [T.ToTensor()]
        if normalize:
            transform_pipelines.append(T.Normalize(mean, std))
        self.transform = T.Compose(transform_pipelines)

    def __call__(self, pil_img: Image.Image) -> torch.Tensor:
        return self.transform(pil_img)


# Default image transform
default_transform = ImageTransform()


def process_image(
    image: Image.Image,
    base_size: int = 1024,
    image_transform: Optional[ImageTransform] = None
) -> torch.Tensor:
    """
    Process a single image for the vision encoder.

    For tier-1 overfitting, we use a simple approach:
    - Pad the image to base_size x base_size while preserving aspect ratio
    - Normalize with mean=0.5, std=0.5

    Args:
        image: PIL Image (RGB)
        base_size: Target size for the image
        image_transform: Optional custom transform

    Returns:
        Tensor of shape (1, 3, base_size, base_size)
    """
    if image_transform is None:
        image_transform = default_transform

    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Pad to square while preserving aspect ratio
    padded = ImageOps.pad(
        image,
        (base_size, base_size),
        color=tuple(int(x * 255) for x in image_transform.mean)
    )

    # Transform to tensor
    tensor = image_transform(padded)

    # Add batch dimension
    return tensor.unsqueeze(0)


def count_vision_tokens(base_size: int = 1024, patch_size: int = 16, downsample_ratio: int = 4) -> int:
    """
    Calculate the number of vision tokens for an image.

    The formula is: (num_queries + 1) * num_queries + 1
    where num_queries = base_size / patch_size / downsample_ratio

    For base_size=1024, patch_size=16, downsample_ratio=4:
    num_queries = 1024 / 16 / 4 = 16
    tokens = (16 + 1) * 16 + 1 = 273

    Args:
        base_size: Image size after padding
        patch_size: Patch size of the vision encoder
        downsample_ratio: Downsampling ratio from SAM compression

    Returns:
        Number of vision tokens
    """
    num_queries = base_size // patch_size // downsample_ratio
    # (num_queries + 1) for each row (including newline token) * num_queries rows + 1 view separator
    return (num_queries + 1) * num_queries + 1


def expand_image_tokens(input_ids: List[int], image_token_id: int, num_tokens: int) -> List[int]:
    """
    Expand single <image> token to N copies.

    Args:
        input_ids: List of token IDs
        image_token_id: The ID of the <image> token
        num_tokens: Number of tokens to expand to

    Returns:
        New list with expanded image tokens
    """
    result = []
    for tok in input_ids:
        if tok == image_token_id:
            result.extend([image_token_id] * num_tokens)
        else:
            result.append(tok)
    return result


if __name__ == "__main__":
    # Test image processing
    from PIL import Image
    import numpy as np

    # Create a test image
    test_img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))

    # Process it
    tensor = process_image(test_img, base_size=1024)
    print(f"Output tensor shape: {tensor.shape}")  # Should be (1, 3, 1024, 1024)
    print(f"Tensor min: {tensor.min():.3f}, max: {tensor.max():.3f}")  # Should be around -1 to 1

    # Count tokens
    n_tokens = count_vision_tokens(base_size=1024)
    print(f"Number of vision tokens for 1024x1024: {n_tokens}")  # Should be 273
