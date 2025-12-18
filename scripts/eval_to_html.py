"""
Evaluation to HTML - Generate HTML report showing model outputs on validation data.

Usage:
    python -m scripts.eval_to_html                           # Use defaults
    python -m scripts.eval_to_html --checkpoint=checkpoints/step_2000.pt
    python -m scripts.eval_to_html --num_samples=20 --max_new_tokens=512
"""

import json
import os
import base64
from io import BytesIO

import torch
from PIL import Image

from nanochat.gpt import GPTConfig
from nanochat.nano_deepseek_ocr import build_nano_deepseek_ocr
from nanochat.image_process import process_image, count_vision_tokens, expand_image_tokens
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import autodetect_device_type

# -----------------------------------------------------------------------------
# Settings
checkpoint = "checkpoints/step_2000.pt"
data_dir = "data/olmocr"
output_html = "data/inference_results.html"
num_samples = 10  # number of samples to evaluate
max_new_tokens = 256
temperature = 0.0  # greedy decoding
base_size = 1024
device_type = ""

# Override from command line
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# Setup
device_type = autodetect_device_type() if device_type == "" else device_type
device = torch.device(device_type)
print(f"Device: {device}")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = RustBPETokenizer.from_directory("tokenizer")
vocab_size = tokenizer.get_vocab_size()
image_token_id = tokenizer.encode_special("<|image|>")
n_img_tokens = count_vision_tokens(base_size)
print(f"Vocab size: {vocab_size:,}, Image tokens: {n_img_tokens}")

# Build model
print("Building model...")
pretrained_vocab_size = 65536
num_layers = 20
model_dim = 1280
num_heads = 16
num_kv_heads = 16
seq_len = 4096

gpt_config = GPTConfig(
    sequence_len=seq_len,
    vocab_size=vocab_size,  # Use expanded vocab size
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
)
model = build_nano_deepseek_ocr(gpt_config=gpt_config)
model.set_image_token_id(image_token_id)

# Load checkpoint
print(f"Loading checkpoint from {checkpoint}...")
state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

# -----------------------------------------------------------------------------
# Load validation data
val_json = os.path.join(data_dir, "val.json")
print(f"Loading validation data from {val_json}...")
with open(val_json, encoding="utf-8") as f:
    samples = json.load(f)

print(f"Found {len(samples)} validation samples, evaluating {num_samples}")

# -----------------------------------------------------------------------------
# Generate outputs
results = []

def generate_safe(model, input_ids, pixel_values, max_new_tokens, temperature, image_token_id):
    """Generate with image token masking to prevent errors."""
    device = input_ids.device

    # Pre-compute vision embeddings once
    vision_embeds = model.encode_images(pixel_values)
    num_vision_tokens = vision_embeds.shape[1]

    # Track original image token positions
    original_len = input_ids.shape[1]

    for _ in range(max_new_tokens):
        # Get text embeddings
        text_embeds = model.gpt.transformer.wte(input_ids)

        # Only replace image tokens in the original prompt region
        if original_len > 0:
            original_mask = input_ids[0, :original_len] == image_token_id
            if original_mask.sum() == num_vision_tokens:
                text_embeds[0, :original_len][original_mask] = vision_embeds[0].to(text_embeds.dtype)

        # Forward through GPT
        logits = model._forward_gpt_with_embeds(text_embeds, targets=None)
        logits = logits[:, -1, :]  # (1, vocab_size)

        # Mask out image token to prevent generating it
        logits[:, image_token_id] = float('-inf')

        if temperature > 0:
            logits = logits / temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_id], dim=1)

        # Stop on EOS (token 0 or 2)
        if next_id.item() in [0, 2]:
            break

    return input_ids


for i, sample in enumerate(samples[:num_samples]):
    print(f"\n[{i+1}/{num_samples}] Processing {sample['id']}...")

    try:
        # Load and process image
        img_path = os.path.join(data_dir, sample["image"])
        image = Image.open(img_path)
        pixel_values = process_image(image, base_size).unsqueeze(0).to(device)

        # Prepare input
        prompt_text = sample["prompt"].replace("<image>", "<|image|>")
        prompt_ids = tokenizer.enc.encode(prompt_text, allowed_special={"<|image|>"})
        expanded = expand_image_tokens(prompt_ids, image_token_id, n_img_tokens)
        input_ids = torch.tensor([expanded], dtype=torch.long, device=device)

        # Generate with safe function
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            with torch.inference_mode():
                output_ids = generate_safe(
                    model, input_ids, pixel_values,
                    max_new_tokens, temperature, image_token_id
                )

        # Decode output (skip input tokens)
        generated_ids = output_ids[0, len(expanded):].tolist()
        generated_text = tokenizer.decode(generated_ids)
    except Exception as e:
        print(f"  Error: {e}")
        generated_text = f"[ERROR: {e}]"

    # Convert image to base64 for HTML embedding
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    results.append({
        "id": sample["id"],
        "image_base64": img_base64,
        "expected": sample["answer"],
        "generated": generated_text,
        "prompt": sample["prompt"],
    })

    print(f"  Expected: {sample['answer'][:100]}...")
    print(f"  Generated: {generated_text[:100]}...")

# -----------------------------------------------------------------------------
# Generate HTML
print(f"\nGenerating HTML report to {output_html}...")

html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>OCR Evaluation Results</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .summary {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .sample {
            background: white;
            border-radius: 8px;
            margin-bottom: 30px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .sample-header {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
            color: #333;
        }
        .content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .image-container {
            text-align: center;
        }
        .image-container img {
            max-width: 100%;
            max-height: 600px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .text-container {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .text-box {
            padding: 15px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-break: break-word;
            overflow-y: auto;
            max-height: 280px;
        }
        .expected {
            background: #e8f5e9;
            border: 1px solid #a5d6a7;
        }
        .generated {
            background: #e3f2fd;
            border: 1px solid #90caf9;
        }
        .label {
            font-weight: bold;
            margin-bottom: 5px;
            color: #555;
        }
        .expected .label { color: #2e7d32; }
        .generated .label { color: #1565c0; }
    </style>
</head>
<body>
    <h1>OCR Evaluation Results</h1>
    <div class="summary">
        <strong>Checkpoint:</strong> """ + checkpoint + """<br>
        <strong>Samples:</strong> """ + str(num_samples) + """<br>
        <strong>Max tokens:</strong> """ + str(max_new_tokens) + """<br>
        <strong>Temperature:</strong> """ + str(temperature) + """
    </div>
"""

for r in results:
    html_content += f"""
    <div class="sample">
        <div class="sample-header">Sample: {r['id']}</div>
        <div class="content">
            <div class="image-container">
                <img src="data:image/jpeg;base64,{r['image_base64']}" alt="{r['id']}">
            </div>
            <div class="text-container">
                <div class="text-box expected">
                    <div class="label">Expected Output:</div>
{r['expected']}
                </div>
                <div class="text-box generated">
                    <div class="label">Generated Output:</div>
{r['generated']}
                </div>
            </div>
        </div>
    </div>
"""

html_content += """
</body>
</html>
"""

with open(output_html, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"HTML report saved to {output_html}")
print("Done!")
