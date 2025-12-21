# vLLM Inference for nano-deepseek-ocr (Sidequest)

This documents how to run efficient inference using DeepSeek-OCR's vLLM implementation with nano-deepseek-ocr checkpoints.

## Reference Code Location

`reference/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/`

## Checkpoint Compatibility

nano-deepseek-ocr checkpoints are **mostly compatible** with vLLM DeepSeek-OCR, but require key renaming.

### Key Mapping

| nano-deepseek-ocr | vLLM Expected | Status |
|-------------------|---------------|--------|
| `sam_model.*` | `sam_model.*` | Direct match |
| `vision_model.*` | `vision_model.*` | Direct match |
| `projector.*` | `projector.*` | Direct match |
| `image_newline` | `image_newline` | Direct match |
| `view_separator` | `view_seperator` | Typo mismatch |
| `gpt.*` | `language.*` | Prefix mismatch |

### Weight Conversion Script

```python
def convert_nanochat_to_vllm(state_dict):
    """Convert nano-deepseek-ocr checkpoint to vLLM format."""
    new_state = {}
    for k, v in state_dict.items():
        # Fix typo in vLLM code
        if k == 'view_separator':
            new_state['view_seperator'] = v
        # Rename GPT prefix to language
        elif k.startswith('gpt.'):
            new_state['language.' + k[4:]] = v
        else:
            new_state[k] = v
    return new_state

# Usage
import torch
state = torch.load('checkpoints/step_1000.pt')
vllm_state = convert_nanochat_to_vllm(state)
torch.save(vllm_state, 'checkpoints/vllm_step_1000.pt')
```

## Alternative: Fix vLLM Reference Code

Instead of converting weights, fix the typo in `deepseek_ocr.py:568`:
- Change `view_seperator` → `view_separator`

## TODO

- [ ] Test converted weights with vLLM inference
- [ ] Benchmark speed comparison vs standard inference
