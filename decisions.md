# Tier-1 Decisions

## PyTorch Version
**Decision**: Use PyTorch 2.4.1+cu121 instead of latest 2.8.0+cu128
**Reason**: CUDA 12.8 binaries require libraries not present in CUDA 12.2 driver environment
**Impact**: Need to manually handle GQA expansion instead of using `enable_gqa` parameter

## Image Token Handling (SUPERSEDED)
**Original Decision**: Use vocab_id 65535 (last token) as image placeholder
**Updated Decision**: Use `<|image|>` special token (vocab_id 65536)
**Reason**: Proper special token handling is cleaner and more maintainable
**Implementation**: Extended RustBPETokenizer.from_directory() to add new special tokens at runtime
**See also**: "Unified Tokenizer" section below

## GQA Compatibility Fix
**Decision**: Manually expand k,v tensors with `repeat_interleave` for GQA
**Reason**: `enable_gqa` parameter not available in PyTorch < 2.5
**Code change**: gpt.py lines 89-93

## Vision Encoder Weights
**Decision**: Load SAM from facebook/sam-vit-base, CLIP from openai/clip-vit-large-patch14
**Reason**: These are the base models used by DeepSeek-OCR
**Note**: net_2 and net_3 compression layers in SAM are randomly initialized (not in HF model)

## Training Configuration
**Decision**: Use AdamW with lr=5e-5, cosine annealing schedule (Stage 1)
**Reason**: DeepSeek-OCR paper recommends AdamW with cosine annealing for Stage 1
**Implementation**: `get_lr()` in vis_tok_train.py uses warmup + cosine decay to 0
**Result**: Achieved near-zero loss (0.007) in 1000 steps

## Model Config (n_kv_head)
**Decision**: Set n_kv_head=16 (same as n_head, no GQA)
**Reason**: Pretrained nanochat base-d20 weights have full-size k,v projections
**Discovery**: Initial config had n_kv_head=4 which caused shape mismatch

## Dependencies Added
- torchvision>=0.17.0 (for image transforms)
- easydict>=1.10 (for CLIP config)
- pillow>=10.0.0 (for image loading)
- safetensors>=0.4.0 (for weight loading)
- transformers>=4.40.0 (for SAM/CLIP pretrained weights)

## Vision Encoder Training - RESOLVED
**Initial concern**: Training with unfrozen vision encoders caused feature collapse at 500 steps
- Before training: SAM cosine sim between images = 0.59 (good)
- After 500 steps: SAM cosine sim = 0.9998 (collapse)

**Resolution**: Longer training with larger batch size RESOLVES the collapse
- 300 steps × batch_size=10 = 3000 sample iterations
- SAM learns image-specific features successfully
- 100% accuracy on all 10 samples

**Final Decision**: Keep all parameters trainable (SAM, CLIP, projector, GPT)
- No need to freeze vision encoders for tier-1
- Sufficient training allows SAM to learn image differentiation
- May revisit freezing for larger-scale training to prevent overfitting

## Batch Size Optimization
**Decision**: Use batch_size=10 for tier-1 (all samples in one batch)
**Reason**: A100 80GB has plenty of memory, only ~25GB used with batch_size=2
**Impact**:
- Throughput: 4.9 samples/s (vs 3.4 with batch_size=2)
- Training time: ~10 minutes for 300 steps
- Better gradient estimation from full-batch updates

## Inference Optimization
**Decision**: Cache vision embeddings during generation
**Reason**: Without caching, each token generation re-encodes the image through SAM+CLIP
**Implementation**: `generate()` method pre-computes vision_embeds once, passes to `forward()`
**Result**: 2.5x faster inference (10+ min → 4 min for 10 samples)

## Stage 2 Training Setup (vis_mid_train.py)
**Decision**: Freeze SAM encoder only (not projector), train CLIP + projector + GPT
**Reason**: Per PLAN.md - Stage 2 freezes "SAM + Conv" (the conv is part of SAM)

**Frozen**:
- SAM encoder (sam_model): ~95M params

**Trainable**:
- CLIP encoder (vision_model): ~303M params
- Projector: ~2.6M params
- GPT (language model): ~560M params
- Special tokens: ~2.5K params
- Total trainable: ~866M params

**Differences from Stage 1**:
- SAM frozen (Stage 1 trains everything)
- lr = 3e-5 (Stage 1 uses 5e-5)
- StepLR decay every 2000 steps (Stage 1 uses constant LR after warmup)
- seq_len = 8192 (Stage 1 uses 4096)

## Text Data Mixing (Stage 2)
**Decision**: Implemented mixed vision + text training via `text_ratio` config
**Implementation** (Karpathy style - no new abstractions):
- Added `text_ratio` config var (default 0.1 = 10% text, 90% vision)
- Reuse existing `tokenizing_distributed_data_loader` for FineWeb text data
- Simple `if/else` in training loop picks data source per step
- When `pixel_values=None`, model skips vision encoder and processes as pure text
- Set `text_ratio=0.0` for vision-only training (tier-1/tier-2 overfitting)

**Rationale**:
- Text mixing prevents catastrophic forgetting of language capabilities
- DeepSeek-OCR paper uses 90% vision + 10% text ratio
- Karpathy style: no wrapper classes, logic inline in training loop

## Unified Tokenizer
**Decision**: Use single tokenizer with `<|image|>` special token for both vision and text
**Implementation**:
- Added `<|image|>` to SPECIAL_TOKENS in tokenizer.py
- Extended vocab from 65536 → 65537 at runtime
- Embedding layers expanded after loading pretrained weights
**Benefit**: Same tokenizer handles vision prompts (with `<|image|>`) and pure text (without)

## Karpathy-Style Code Refactoring
**Decision**: Refactor all vision training code to match Karpathy's nanochat patterns
**Reason**: Consistency, readability, and maintainability

**Patterns adopted**:
- Config variables at top of file, overridable via configurator.py
- Generator functions instead of Dataset/DataLoader classes
- `split` parameter for train/val distinction
- `build_val_loader = lambda: ...` pattern for fresh val loaders
- EMA-smoothed loss logging
- Inline, flat code instead of class abstractions

**Files refactored**:
- `image_process.py`: 170 → 66 lines (removed classes)
- `vision_dataloader.py`: Rewritten with `split` parameter
- `vis_tok_train.py`: Config-at-top pattern
- `vis_mid_train.py`: Matching Stage 2 script
- `vision_sample.py`: 411 → 148 lines (reuse modules)

## Multi-GPU Training (DDP)

### DDP Wrapper vs Custom Optimizers
**Decision**: Use standard PyTorch `DistributedDataParallel` wrapper instead of Karpathy's custom optimizers
**Reason**: Karpathy's `DistMuon`/`DistAdamW` are LLM-specific:
- `DistMuon` uses Newton-Schulz orthogonalization for 2D matrix params
- `DistAdamW` requires `param.shape[0] % world_size == 0`
- Vision encoders (convolutions, layernorms) don't satisfy these constraints

**Implementation**:
- Wrap model with `torch.nn.parallel.DistributedDataParallel`
- Use `DistributedSampler` for data sharding
- Guard checkpoint saves with `if master_process:`
- Aggregate validation loss with `dist.all_reduce()`

### CLIP patch_embedding Unused Parameter
**Decision**: Mark `vision_model.embeddings.patch_embedding` as `requires_grad_(False)`
**Reason**: In our architecture, CLIP receives SAM features as patch embeddings instead of computing its own. The patch_embedding conv is never used.
**Code**: `nano_deepseek_ocr.py:83`
```python
self.vision_model.embeddings.patch_embedding.requires_grad_(False)
```

### Mixed Training DDP Flag
**Decision**: Use `find_unused_parameters=(text_ratio > 0)` for vis_mid_train
**Reason**: With mixed vision + text training:
- Vision batches: All params used (vision encoder + GPT)
- Text batches: Vision encoder unused (only GPT)

DDP requires all parameters to receive gradients by default. When `text_ratio > 0`, some steps skip the vision encoder, so we need `find_unused_parameters=True`.

**Code**: `vis_mid_train.py:174-178`
```python
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[ddp_local_rank],
        find_unused_parameters=(text_ratio > 0)
    )
```

**Note**: When `text_ratio=0` (vision-only), no flag needed for better performance.

### Tokenizer Location for Text Data
**Decision**: Copy tokenizer to `~/.cache/nanochat/tokenizer/`
**Reason**: Text dataloader (`tokenizing_distributed_data_loader`) uses hardcoded path
**Impact**: Need to ensure tokenizer exists at both locations:
- `tokenizer/` (project directory, for vision training)
- `~/.cache/nanochat/tokenizer/` (for text data loader)

## DeepEncoder-only Checkpointing (Stage 1 → Stage 2)

**Decision**: Save DeepEncoder-only checkpoint at end of Stage 1, require it for Stage 2
**Reason**: DeepSeek-OCR paper discards decoder after Stage 1 training
**Implementation**:
- `vis_tok_train.py`: Saves `deepencoder_{steps}.pt` (SAM + CLIP + projector + special tokens, excludes GPT)
- `vis_mid_train.py`: REQUIRES `--resume_from_deepencoder=<path>`, loads trained DeepEncoder + fresh nanochat GPT

**Checkpoint files**:
| File | Contents | Purpose |
|------|----------|---------|
| `step_{N}.pt` | Full model | Resume Stage 1, debugging |
| `deepencoder_{N}.pt` | Vision encoder only | Stage 2 input |

**Usage**:
```bash
# Stage 1
python -m scripts.vis_tok_train --steps=500
# Creates: checkpoints/deepencoder_500.pt

# Stage 2
python -m scripts.vis_mid_train --resume_from_deepencoder=checkpoints/deepencoder_500.pt
```

**Reference**: DeepSeek-OCR paper Section 3.2 - "After DeepEncoder is ready, we use data mentioned in Section 3.4 to train the DeepSeek-OCR" (implying decoder is replaced)

## LLaVA-Style Two-Phase Training (2024-12-31)

**Problem**: Full SAM + CLIP training caused model collapse in Stage 1
- SAM cosine similarity between images went from 0.59 → 0.9998 (complete collapse)
- Model memorized outputs by prompt category, ignored image content
- Root cause: Insufficient data diversity (380K images vs DeepSeek-OCR's 30M)

**Decision**: Adopt LLaVA 1.5 style two-phase training

### Phase 1: Alignment (vis_tok_train.py)
**Train**: net_2, net_3 (compression conv adapter) + projector (~8.5M params)
**Freeze**: SAM core (patch_embed, pos_embed, blocks, neck), CLIP, GPT

**Hyperparameters** (LLaVA Stage 1):
- LR: 1e-3 (high for small adapter training)
- Batch: 256
- Epochs: 1
- Warmup: 3% of total steps (warmup_ratio=0.03)
- LR schedule: Cosine annealing to 0
- Data: LLaVA_Instruct_150K

**Rationale**:
- net_2 and net_3 are DeepSeek-OCR additions (not pretrained) - must be trained
- SAM core is pretrained on 1B masks - freezing preserves visual features
- High LR (1e-3) is safe for small adapter training (LLaVA recommendation)

### Phase 2: Fine-tuning (vis_mid_train.py)
**Train**: projector + GPT (~564M params)
**Freeze**: ALL SAM (including net_2, net_3), CLIP

**Hyperparameters** (LLaVA Stage 2):
- LR: 2e-5 (lower for full model fine-tuning)
- Batch: 128
- Epochs: 1
- Warmup: 3% of total steps (warmup_ratio=0.03)
- LR schedule: Cosine annealing to 0
- Data: 70% vision (olmOCR + LLaVA_Instruct) + 30% text (SmolTalk + MMLU + GSM8K)

**Rationale**:
- Vision tokenizer (SAM + conv + CLIP) is now frozen to preserve learned representations
- Lower LR for fine-tuning full model (projector + GPT)
- Mixed vision + text prevents catastrophic forgetting of language capabilities

### Key Differences from DeepSeek-OCR
| Aspect | DeepSeek-OCR | Our Approach |
|--------|--------------|--------------|
| Stage 1 | Trains full SAM + CLIP | Freeze SAM core, train only net_2/net_3 |
| Data scale | 30M images | ~150K images |
| Stage 2 | Freezes SAM + compressor | Freezes ALL vision (SAM + net_2/net_3 + CLIP) |

**Why this works with limited data**:
- Pretrained SAM/CLIP features are preserved (no collapse)
- Only ~8.5M params trained in Phase 1 (vs 962M in original approach)
- Phase 2 focuses on language model adaptation with frozen vision

### Architecture Reference
```
SAM ViT-B (FREEZE in both phases)
├── patch_embed
├── pos_embed
├── blocks[12]
└── neck (768→256)

DeepSeek Adapter (TRAIN Phase 1, FREEZE Phase 2)
├── net_2: Conv2d(256→512)
└── net_3: Conv2d(512→1024)

CLIP ViT-L (FREEZE in both phases)

Projector (TRAIN both phases)
└── Linear(2048→1280)

GPT (FREEZE Phase 1, TRAIN Phase 2)
```
