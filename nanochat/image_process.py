# Copied from https://github.com/deepseek-ai/DeepSeek-OCR
# Modifications:
# - Simplified for nanochat integration
# - Removed VLLM-specific code
# - Removed ImageTransform and ImageProcessor classes (Karpathy style: just functions)

"""Image processing utilities for nano-deepseek-ocr."""

from typing import List
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps

# Normalization constants
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
_transform = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


def process_image(image: Image.Image, base_size: int = 1024) -> torch.Tensor:
    """
    Pad image to square, normalize, return (3, H, W) tensor.

    Args:
        image: PIL Image
        base_size: Target size (default 1024)

    Returns:
        Tensor of shape (3, base_size, base_size)
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    padded = ImageOps.pad(image, (base_size, base_size),
                          color=tuple(int(x * 255) for x in MEAN))
    return _transform(padded)


def count_vision_tokens(base_size: int = 1024, patch_size: int = 16, downsample_ratio: int = 4) -> int:
    """
    Calculate vision token count: (nq + 1) * nq + 1

    For base_size=1024: nq = 1024/16/4 = 16, tokens = 17*16+1 = 273
    """
    nq = base_size // patch_size // downsample_ratio
    return (nq + 1) * nq + 1


def expand_image_tokens(input_ids: List[int], image_token_id: int, num_tokens: int) -> List[int]:
    """Expand single <image> token to num_tokens copies."""
    result = []
    for tok in input_ids:
        if tok == image_token_id:
            result.extend([image_token_id] * num_tokens)
        else:
            result.append(tok)
    return result


if __name__ == "__main__":
    import numpy as np
    test_img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    tensor = process_image(test_img, base_size=1024)
    print(f"Shape: {tensor.shape}")  # (3, 1024, 1024)
    print(f"Range: [{tensor.min():.2f}, {tensor.max():.2f}]")  # ~[-1, 1]
    print(f"Vision tokens: {count_vision_tokens()}")  # 273
