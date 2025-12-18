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
**Decision**: Use AdamW with lr=1e-4, cosine annealing schedule
**Reason**: DeepSeek-OCR paper recommends AdamW for vision-language alignment
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
**Decision**: Skip text data mixing for tier-1/tier-2 testing
**Reason**:
- Text mixing (90% vision + 10% text) is for preventing catastrophic forgetting during full-scale training
- For tier-1/tier-2 overfitting tests, vision-only data is sufficient
- The unified tokenizer with `<|image|>` can handle both vision and text data
- When pixel_values=None, model skips vision encoder and processes as pure text

**Future**: Add text mixing when scaling to tier-3 with larger datasets

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
