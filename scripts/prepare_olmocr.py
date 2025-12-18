"""
Prepare olmOCR dataset for vision training.

Downloads PDF pages from HuggingFace olmOCR-mix-1025 dataset,
converts them to images, and creates train.json/val.json in Karpathy format.

Usage:
    python -m scripts.prepare_olmocr --num_train=300 --num_val=10
"""

import json
import os
import tarfile
import tempfile
from io import BytesIO

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from pdf2image import convert_from_bytes
from PIL import Image

# -----------------------------------------------------------------------------
# Config
num_train = 300
num_val = 10
output_dir = "data/olmocr"
config_name = "00_documents"  # documents config has most variety

# Override from command line
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())

# -----------------------------------------------------------------------------
print(f"Preparing olmOCR dataset: {num_train} train, {num_val} val samples")
print(f"Output directory: {output_dir}")

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

# -----------------------------------------------------------------------------
# Load dataset metadata from HuggingFace
print(f"\nLoading dataset metadata from allenai/olmOCR-mix-1025 ({config_name})...")
ds = load_dataset("allenai/olmOCR-mix-1025", name=config_name, split="train", streaming=True)

# Collect samples (need more than we want to account for failures)
total_needed = num_train + num_val
buffer_size = int(total_needed * 1.5)  # 50% buffer for failures
print(f"Collecting {buffer_size} samples from stream...")

samples = []
for i, sample in enumerate(ds):
    if i >= buffer_size:
        break
    # Filter: English, valid rotation, not too long text
    if (sample.get("primary_language") == "en" and
        sample.get("is_rotation_valid", True) and
        len(sample.get("natural_text", "")) > 50 and  # minimum text length
        len(sample.get("natural_text", "")) < 10000):  # max reasonable length
        samples.append(sample)
    if len(samples) >= total_needed:
        break
    if (i + 1) % 100 == 0:
        print(f"  Scanned {i + 1} samples, collected {len(samples)}")

print(f"Collected {len(samples)} valid samples")

if len(samples) < total_needed:
    print(f"Warning: Only found {len(samples)} samples, needed {total_needed}")
    num_train = int(len(samples) * 0.97)
    num_val = len(samples) - num_train

# -----------------------------------------------------------------------------
# Group samples by tarball for efficient downloading
print("\nGrouping samples by tarball...")
tarball_to_samples = {}
for idx, sample in enumerate(samples):
    pdf_relpath = sample.get("pdf_relpath", "")
    if ":" in pdf_relpath:
        tarball_path, file_in_tar = pdf_relpath.split(":", 1)
        if tarball_path not in tarball_to_samples:
            tarball_to_samples[tarball_path] = []
        tarball_to_samples[tarball_path].append((idx, file_in_tar, sample))

print(f"Samples span {len(tarball_to_samples)} tarballs")

# -----------------------------------------------------------------------------
# Download tarballs and extract PDFs
print("\nDownloading tarballs and converting PDFs to images...")

processed_samples = []
total_to_process = min(total_needed, len(samples))

for tarball_path, tarball_samples in tarball_to_samples.items():
    if len(processed_samples) >= total_to_process:
        break

    print(f"\n  Downloading {tarball_path}...")
    try:
        local_tar = hf_hub_download(
            repo_id="allenai/olmOCR-mix-1025",
            filename=tarball_path,
            repo_type="dataset",
        )
    except Exception as e:
        print(f"  Failed to download {tarball_path}: {e}")
        continue

    print(f"  Extracting PDFs and converting to images...")
    with tarfile.open(local_tar, "r:gz") as tar:
        for idx, file_in_tar, sample in tarball_samples:
            if len(processed_samples) >= total_to_process:
                break

            try:
                # Extract PDF bytes
                member = tar.getmember(file_in_tar)
                pdf_bytes = tar.extractfile(member).read()

                # Convert first page to image
                images = convert_from_bytes(pdf_bytes, first_page=1, last_page=1, dpi=150)
                if not images:
                    continue

                img = images[0]

                # Convert to RGB if needed
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Save image
                img_filename = f"olm_{len(processed_samples):05d}.jpg"
                img_path = os.path.join(output_dir, "images", img_filename)
                img.save(img_path, "JPEG", quality=90)

                # Create sample entry with type-specific prompt
                text = sample.get("natural_text", "").strip()
                is_table = sample.get("is_table", False)
                is_diagram = sample.get("is_diagram", False)

                # Use different prompts based on content type (DeepSeek-OCR style)
                if is_table:
                    prompt = "<image>\nConvert this table to HTML."
                    doc_type = "table"
                elif is_diagram:
                    prompt = "<image>\nDescribe this diagram."
                    doc_type = "diagram"
                else:
                    prompt = "<image>\nOCR this document."
                    doc_type = "document"

                processed_samples.append({
                    "id": f"olm_{len(processed_samples):05d}",
                    "image": f"images/{img_filename}",
                    "source": "olmOCR",
                    "type": doc_type,
                    "prompt": prompt,
                    "answer": text,
                    "metadata": {
                        "url": sample.get("url", ""),
                        "page": sample.get("page_number", 1),
                        "language": sample.get("primary_language", "en"),
                        "is_table": is_table,
                        "is_diagram": is_diagram,
                    }
                })

                if len(processed_samples) % 50 == 0:
                    print(f"    Processed {len(processed_samples)}/{total_to_process} samples")

            except Exception as e:
                print(f"    Failed to process {file_in_tar}: {e}")
                continue

print(f"\nTotal processed: {len(processed_samples)} samples")

# -----------------------------------------------------------------------------
# Split into train/val
train_samples = processed_samples[:num_train]
val_samples = processed_samples[num_train:num_train + num_val]

print(f"\nSplit: {len(train_samples)} train, {len(val_samples)} val")

# -----------------------------------------------------------------------------
# Save JSON files
train_path = os.path.join(output_dir, "train.json")
val_path = os.path.join(output_dir, "val.json")

with open(train_path, "w", encoding="utf-8") as f:
    json.dump(train_samples, f, indent=2, ensure_ascii=False)
print(f"Saved {train_path}")

with open(val_path, "w", encoding="utf-8") as f:
    json.dump(val_samples, f, indent=2, ensure_ascii=False)
print(f"Saved {val_path}")

# -----------------------------------------------------------------------------
# Print sample info
print("\n" + "="*60)
print("Sample data:")
print("="*60)
if train_samples:
    s = train_samples[0]
    print(f"ID: {s['id']}")
    print(f"Image: {s['image']}")
    print(f"Prompt: {s['prompt']}")
    print(f"Answer length: {len(s['answer'])} chars")
    print(f"Answer preview: {s['answer'][:200]}...")

print("\n" + "="*60)
print("Done! Dataset ready at:", output_dir)
print("="*60)
