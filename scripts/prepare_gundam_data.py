"""
Prepare vlm-gundam10 dataset with high-resolution images (>2000px).

Downloads samples from:
- DocVQA: High-res document images (for PDF/document OCR)
- ChartQA: High-res chart images (fallback to upscaling if needed)
- TextVQA: High-res scene text images

Usage:
    python -m scripts.prepare_gundam_data
"""

import os
import json
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_DIR = "data/gundam_samples"
MIN_RESOLUTION = 2000  # Minimum pixels on longest edge


def get_max_dimension(img):
    """Get the longest edge of an image."""
    return max(img.size)


def save_image(img, path):
    """Save image, converting to RGB if needed."""
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    img.save(path, quality=95)


def collect_docvqa_samples(num_samples=4):
    """Collect high-resolution document images from DocVQA."""
    print(f"\n[DocVQA] Loading dataset and finding {num_samples} unique high-res samples...")

    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")

    samples = []
    seen_hashes = set()  # Track unique images

    for idx, item in enumerate(tqdm(ds, desc="Scanning DocVQA")):
        img = item['image']
        max_dim = get_max_dimension(img)

        # Create fingerprint to detect duplicates
        import hashlib
        img_bytes = img.tobytes()[:1000]
        img_hash = hashlib.md5(img_bytes).hexdigest()

        if max_dim >= MIN_RESOLUTION and img_hash not in seen_hashes:
            seen_hashes.add(img_hash)
            sample = {
                "id": f"gundam_doc_{len(samples):03d}",
                "image": f"images/gundam_doc_{len(samples):03d}.jpg",
                "source": "DocVQA",
                "type": "document_ocr",
                "resolution": f"{img.size[0]}x{img.size[1]}",
                "prompt": "<image>\n<|grounding|>OCR this document.",
                "question": item.get('question', ''),
                "answer": item.get('answer', item.get('answers', [''])[0] if isinstance(item.get('answers'), list) else ''),
                "_pil_image": img,
            }
            samples.append(sample)
            print(f"  Found: {max_dim}px - {sample['id']}")

            if len(samples) >= num_samples:
                break

    print(f"[DocVQA] Collected {len(samples)} unique samples")
    return samples


def collect_chartqa_samples(num_samples=3):
    """Collect chart images from ChartQA, upscale if needed."""
    print(f"\n[ChartQA] Loading dataset and finding {num_samples} unique samples...")

    ds = load_dataset("HuggingFaceM4/ChartQA", split="train")

    samples = []
    seen_hashes = set()  # Track unique images

    for idx, item in enumerate(tqdm(ds, desc="Scanning ChartQA")):
        img = item['image']
        max_dim = get_max_dimension(img)

        # Create fingerprint to detect duplicates
        import hashlib
        img_bytes = img.tobytes()[:1000]
        img_hash = hashlib.md5(img_bytes).hexdigest()

        # Prefer high-res, but accept lower res charts (we can upscale or use as-is)
        if max_dim >= 800 and img_hash not in seen_hashes:  # Charts are often smaller, accept reasonable sizes
            seen_hashes.add(img_hash)
            # For Gundam mode testing, we can upscale charts to force cropping
            if max_dim < MIN_RESOLUTION:
                scale = MIN_RESOLUTION / max_dim * 1.5  # Upscale to ~3000px
                new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
                img = img.resize(new_size, Image.LANCZOS)
                resolution_note = f"{new_size[0]}x{new_size[1]} (upscaled from {item['image'].size})"
            else:
                resolution_note = f"{img.size[0]}x{img.size[1]}"

            sample = {
                "id": f"gundam_chart_{len(samples):03d}",
                "image": f"images/gundam_chart_{len(samples):03d}.png",
                "source": "ChartQA",
                "type": "chart_qa",
                "resolution": resolution_note,
                "prompt": "<image>\nParse this chart in detail.",
                "question": item.get('query', ''),
                "answer": item.get('label', ''),
                "_pil_image": img,
            }
            samples.append(sample)
            print(f"  Found: {resolution_note} - {sample['id']}")

            if len(samples) >= num_samples:
                break

    print(f"[ChartQA] Collected {len(samples)} unique samples")
    return samples


def collect_textvqa_samples(num_samples=3):
    """Collect high-resolution scene text images from InfographicVQA (part of DocVQA)."""
    print(f"\n[InfographicVQA] Loading dataset and finding {num_samples} unique high-res samples...")

    # Use InfographicVQA which has high-res infographic images with scene text
    ds = load_dataset("lmms-lab/DocVQA", "InfographicVQA", split="validation")

    samples = []
    seen_sizes = set()  # Track unique images by (width, height, first_100_bytes_hash)

    for idx, item in enumerate(tqdm(ds, desc="Scanning InfographicVQA")):
        img = item['image']
        max_dim = get_max_dimension(img)

        # Create a simple fingerprint to detect duplicates
        import hashlib
        img_bytes = img.tobytes()[:1000]  # First 1000 bytes as fingerprint
        img_hash = hashlib.md5(img_bytes).hexdigest()

        # InfographicVQA has varied resolutions, prefer larger ones
        if max_dim >= MIN_RESOLUTION and img_hash not in seen_sizes:
            seen_sizes.add(img_hash)
            resolution_note = f"{img.size[0]}x{img.size[1]}"

            sample = {
                "id": f"gundam_infographic_{len(samples):03d}",
                "image": f"images/gundam_infographic_{len(samples):03d}.jpg",
                "source": "InfographicVQA",
                "type": "infographic_ocr",
                "resolution": resolution_note,
                "prompt": "<image>\n<|grounding|>OCR this infographic.",
                "question": item.get('question', ''),
                "answer": item.get('answers', [''])[0] if item.get('answers') else '',
                "_pil_image": img,
            }
            samples.append(sample)
            print(f"  Found: {max_dim}px - {sample['id']}")

            if len(samples) >= num_samples:
                break

    print(f"[InfographicVQA] Collected {len(samples)} unique samples")
    return samples


def main():
    os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)

    # Collect samples from each source
    all_samples = []

    # 4 document images from DocVQA
    doc_samples = collect_docvqa_samples(num_samples=4)
    all_samples.extend(doc_samples)

    # 3 chart images from ChartQA
    chart_samples = collect_chartqa_samples(num_samples=3)
    all_samples.extend(chart_samples)

    # 3 scene text images from TextVQA
    scene_samples = collect_textvqa_samples(num_samples=3)
    all_samples.extend(scene_samples)

    # Save images and create metadata
    print(f"\n[Saving] Writing {len(all_samples)} samples to {OUTPUT_DIR}/")

    train_data = []
    for sample in all_samples:
        # Save image
        img = sample.pop('_pil_image')
        img_path = f"{OUTPUT_DIR}/{sample['image']}"
        save_image(img, img_path)

        # Verify saved image dimensions
        with Image.open(img_path) as saved_img:
            sample['saved_resolution'] = f"{saved_img.size[0]}x{saved_img.size[1]}"

        train_data.append(sample)
        print(f"  Saved: {sample['id']} ({sample['saved_resolution']})")

    # Write train.json
    with open(f"{OUTPUT_DIR}/train.json", 'w') as f:
        json.dump(train_data, f, indent=2)

    # Also create val.json (same as train for overfitting test)
    with open(f"{OUTPUT_DIR}/val.json", 'w') as f:
        json.dump(train_data, f, indent=2)

    print(f"\n[Done] Dataset prepared:")
    print(f"  - {OUTPUT_DIR}/train.json: {len(train_data)} samples")
    print(f"  - {OUTPUT_DIR}/val.json: {len(train_data)} samples")
    print(f"  - {OUTPUT_DIR}/images/: {len(train_data)} images")

    # Summary
    print("\n[Summary]")
    for sample in train_data:
        print(f"  {sample['id']}: {sample['type']} ({sample['saved_resolution']})")


if __name__ == "__main__":
    main()
