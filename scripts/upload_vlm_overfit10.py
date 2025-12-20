"""
Upload vlm-overfit10 to HuggingFace using ImageFolder format.

Usage:
    python -m scripts.upload_vlm_overfit10 --repo_id=USERNAME/vlm-overfit10

This creates a dataset with the scalable ImageFolder format:
- Images stored as separate files (not embedded in Parquet)
- metadata.jsonl with file_name field linking to images
- Standard format that scales to millions of images
"""

import json
import os
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True, help="HuggingFace repo ID (e.g. username/vlm-overfit10)")
    parser.add_argument("--private", action="store_true", help="Make dataset private")
    args = parser.parse_args()

    # Prepare upload directory structure
    upload_dir = "vlm-overfit10-upload"
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)
    os.makedirs(f"{upload_dir}/train/images", exist_ok=True)

    # Load local data
    data_dir = "data/overfit_samples"
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
                "prompt": s["prompt"],
                "answer": s["answer"],
            }
            f.write(json.dumps(meta) + "\n")

    # Create dataset card README
    readme_content = """---
license: cc-by-4.0
task_categories:
  - image-to-text
  - visual-question-answering
language:
  - en
size_categories:
  - n<1K
---

# VLM Overfit 10

A small dataset of 10 vision samples for VLM (Vision Language Model) validation and overfitting tests.

## Purpose

This dataset is designed for:
- Quick validation of vision encoder training
- Overfitting tests before scaling to larger datasets
- Debugging multimodal training pipelines

## Data Sources

All samples are from established research datasets:

| Source | Count | Type | Description |
|--------|-------|------|-------------|
| SROIE | 4 | Receipt OCR | Malaysian receipts with OCR ground truth |
| ChartQA | 3 | Chart QA | Line/bar chart understanding |
| TextVQA | 3 | Scene Text | Scene text recognition and description |

## Format

Uses HuggingFace ImageFolder format with metadata.jsonl:
- `file_name`: Path to image file
- `id`: Unique sample identifier
- `source`: Original dataset (SROIE, ChartQA, TextVQA)
- `type`: Task type (receipt_ocr, chart_qa, scene_text_vqa)
- `prompt`: Input prompt with `<image>` placeholder
- `answer`: Expected output

## Usage

```python
from datasets import load_dataset

ds = load_dataset("USERNAME/vlm-overfit10", split="train")
print(ds[0])
# {'image': <PIL.Image>, 'id': 'receipt_000', 'source': 'SROIE', ...}
```

## License

Samples derived from:
- SROIE: Academic research dataset
- ChartQA (HuggingFaceM4/ChartQA): CC-BY
- TextVQA: Research dataset

Please check original dataset licenses for commercial use.
"""
    readme_content = readme_content.replace("USERNAME/vlm-overfit10", args.repo_id)
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
