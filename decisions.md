# Tier-1 Decisions

## PyTorch Version
**Decision**: Use PyTorch 2.4.1+cu121 instead of latest 2.8.0+cu128
**Reason**: CUDA 12.8 binaries require libraries not present in CUDA 12.2 driver environment
**Impact**: Need to manually handle GQA expansion instead of using `enable_gqa` parameter

## Image Token Handling
**Decision**: Use vocab_id 65535 (last token) as image placeholder
**Reason**: RustBPETokenizer doesn't support adding special tokens after initialization
**Alternative considered**: Modify tokenizer training - rejected as too complex for tier-1

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
**Decision**: Freeze SAM encoder and projector, train CLIP + GPT
**Frozen**:
- SAM encoder (sam_model): 95.5M params
- Projector (linear 2048→1280): 2.6M params
- Total frozen: 98M params

**Trainable**:
- CLIP encoder (vision_model): 303M params
- GPT (language model): 560M params
- Special tokens (image_newline, view_separator): 2.5K params
- Total trainable: 864M params (89.8%)

**Result**: Stage 2 verified working - converges quickly when starting from Stage 1 checkpoint
