"""
Upload vlm-gundam10 to HuggingFace using ImageFolder format.

Usage:
    python -m scripts.upload_vlm_gundam10 --repo_id=USERNAME/vlm-gundam10

This creates a dataset with high-resolution images for Gundam mode testing:
- Images >2000px to force multi-resolution dynamic cropping
- Document OCR, chart parsing, and infographic analysis tasks
"""

import json
import os
import shutil
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True, help="HuggingFace repo ID (e.g. username/vlm-gundam10)")
    parser.add_argument("--private", action="store_true", help="Make dataset private")
    args = parser.parse_args()

    # Prepare upload directory structure
    upload_dir = "vlm-gundam10-upload"
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)
    os.makedirs(f"{upload_dir}/train/images", exist_ok=True)

    # Load local data
    data_dir = "data/gundam_samples"
    with open(f"{data_dir}/train.json") as f:
        samples = json.load(f)

    print(f"Processing {len(samples)} samples...")

    # Write metadata.jsonl and copy images
    with open(f"{upload_dir}/train/metadata.jsonl", "w") as f:
        for s in samples:
            # Copy image to upload dir
            src = f"{data_dir}/{s['image']}"
            dst = f"{upload_dir}/train/{s['image']}"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)

            # Write metadata line (ImageFolder format)
            meta = {
                "file_name": s["image"],  # relative path to image
                "id": s["id"],
                "source": s["source"],
                "type": s["type"],
                "resolution": s.get("saved_resolution", s.get("resolution", "")),
                "prompt": s["prompt"],
                "answer": s["answer"] if isinstance(s["answer"], str) else s["answer"][0],
            }
            f.write(json.dumps(meta) + "\n")

    # Create dataset card README
    readme_content = """---
license: cc-by-4.0
task_categories:
  - image-to-text
  - visual-question-answering
  - document-question-answering
language:
  - en
size_categories:
  - n<1K
tags:
  - gundam-mode
  - high-resolution
  - ocr
  - document-understanding
---

# VLM Gundam 10

A dataset of 10 **high-resolution** vision samples for testing Gundam mode (multi-resolution dynamic cropping).

## Purpose

This dataset is designed for:
- Testing multi-resolution vision encoding (Gundam mode)
- Validating dynamic crop processing for images >2000px
- Overfitting tests on high-resolution documents, charts, and infographics

## What is Gundam Mode?

Gundam mode processes high-resolution images using:
1. **Global view**: 1024×1024 → 273 tokens
2. **Local crops**: 640×640 each → 100 tokens per crop
3. Dynamic tile selection based on aspect ratio

This allows the model to capture both overall layout and fine-grained details.

## Data Sources

| Source | Count | Type | Resolution Range | Description |
|--------|-------|------|------------------|-------------|
| DocVQA | 4 | Document OCR | 2257-2386px | High-res scanned documents |
| ChartQA | 3 | Chart QA | 2290-3000px | Upscaled chart images |
| InfographicVQA | 3 | Infographic OCR | 2221-8256px | Long infographics |

## Resolution Distribution

All images have longest edge >2000px to force Gundam mode:
- Minimum: 2221px
- Maximum: **8256px** (very long infographic)
- Average: ~3000px

## Format

Uses HuggingFace ImageFolder format with metadata.jsonl:
- `file_name`: Path to image file
- `id`: Unique sample identifier
- `source`: Original dataset (DocVQA, ChartQA, InfographicVQA)
- `type`: Task type (document_ocr, chart_qa, infographic_ocr)
- `resolution`: Image dimensions (WxH)
- `prompt`: Input prompt with `<image>` placeholder
- `answer`: Expected output

## Usage

```python
from datasets import load_dataset

ds = load_dataset("USERNAME/vlm-gundam10", split="train")
print(ds[0])
# {'image': <PIL.Image>, 'id': 'gundam_doc_000', 'resolution': '2257x1764', ...}
```

## License

Samples derived from:
- DocVQA: UCSF Industry Documents Library
- ChartQA (HuggingFaceM4/ChartQA): CC-BY
- InfographicVQA: Research dataset

Please check original dataset licenses for commercial use.
"""
    readme_content = readme_content.replace("USERNAME/vlm-gundam10", args.repo_id)
    with open(f"{upload_dir}/README.md", "w") as f:
        f.write(readme_content)

    print(f"Prepared {upload_dir}/ for upload")
    print(f"  - train/metadata.jsonl: {len(samples)} entries")
    print(f"  - train/images/: {len(samples)} images")
    print(f"  - README.md: dataset card")

    # Upload to HuggingFace
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True, private=args.private)
    api.upload_folder(
        folder_path=upload_dir,
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    print(f"\nUploaded to https://huggingface.co/datasets/{args.repo_id}")

    # Cleanup
    shutil.rmtree(upload_dir)
    print(f"Cleaned up {upload_dir}/")


if __name__ == "__main__":
    main()
