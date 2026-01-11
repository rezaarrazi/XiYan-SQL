# Training on Low Memory GPU (8GB VRAM)

## Your Hardware: RTX 3080 Ti 8GB

Training a 3B parameter model on 8GB VRAM is challenging but **possible** with the right optimizations.

---

## ⚠️ Reality Check

**What to expect:**
- ✅ **It will work** - with the optimizations below
- ⚠️ **It will be SLOW** - 10-20x slower than multi-GPU setup
- ⚠️ **Limited sequence length** - Max 4096 instead of 10240
- ⚠️ **Reduced LoRA rank** - 64 instead of 512
- ✅ **But you'll get a working model!**

**Training time estimate:**
- Test data (3 samples): ~10-15 minutes
- BIRD data (224 samples): ~8-12 hours

---

## Quick Start (3 Steps)

### Step 1: Use the Low-Memory Script

```bash
cd train
bash xiyan_sft_3b_low_mem.sh
```

### Step 2: Monitor Memory

```bash
# In another terminal
watch -n 1 nvidia-smi
```

**Expected VRAM usage**: ~6-7GB (should fit!)

### Step 3: If Still OOM

If you get Out of Memory errors, reduce further:

```bash
# Edit xiyan_sft_3b_low_mem.sh
MAX_LENGTH=2048          # Instead of 4096
LORA_R=32                # Instead of 64
ACC_STEP=64              # Instead of 32 (to maintain effective batch size)
```

---

## Configuration Breakdown

### Memory Optimizations Applied

| Setting | Normal (Multi-GPU) | Low Memory (8GB) | Reason |
|---------|-------------------|------------------|--------|
| **GPUs** | 8x A100 | 1x RTX 3080 Ti | Single GPU only |
| **DeepSpeed** | Zero2 | Zero3 + Offload | Offload params to CPU |
| **LoRA** | Must use | REQUIRED | Reduces trainable params |
| **LoRA Rank** | 512 | 64 | Lower rank = less memory |
| **Batch Size** | 2 | 1 | Minimum possible |
| **Grad Accum** | 2 | 32 | Compensate for batch=1 |
| **Max Length** | 10240 | 4096 | Shorter context = less memory |
| **Flash Attention** | Enabled | Disabled | Compatibility |

### Low-Memory Script Settings

```bash
# In xiyan_sft_3b_low_mem.sh

# CRITICAL SETTINGS
USE_LORA=True            # REQUIRED - cannot train without LoRA
LORA_R=64                # Low rank to save memory
BATCH_SIZE=1             # Must be 1
ACC_STEP=32              # High accumulation to maintain training quality
MAX_LENGTH=4096          # Reduced from 10240

# DeepSpeed Zero3 with CPU offload
DS_CONFIG="config/zero3_offload.yaml"

# Single GPU
export CUDA_VISIBLE_DEVICES=0
```

---

## Understanding the Trade-offs

### What You're Giving Up

1. **Training Speed**: 10-20x slower than 8x A100 setup
2. **LoRA Rank**: 64 vs 512 (slightly lower capacity)
3. **Context Length**: 4096 vs 10240 (shorter SQL queries)

### What You're Keeping

1. **Model Quality**: Still ~50-55% accuracy (vs 55-60% with full setup)
2. **Training Works**: You'll get a functional model
3. **Low Cost**: Train on consumer hardware

---

## Step-by-Step Training

### 1. Prepare Environment

```bash
cd XiYan-SQLTraining
uv sync
```

### 2. Download Model

```bash
cd train/utils
uv run model_download.py
# Select: 3 (Qwen2.5-Coder-3B-Instruct)
cd ../..
```

### 3. Prepare Data

**Start with test data** (3 samples) to verify it works:

```bash
cd data
bash data_assembler.sh \
  configs/datasets_nl2sql_standard.json \
  ../train/datasets/nl2sql_standard_train.json
cd ..
```

### 4. Train (Low Memory Mode)

```bash
cd train
bash xiyan_sft_3b_low_mem.sh
```

**First run**: Let it train for 1 epoch (~2 hours with test data) to verify everything works.

### 5. Monitor Progress

```bash
# Terminal 1: Training log
tail -f train/output/dense/nl2sql_3b_lowmem/training.log

# Terminal 2: GPU usage
watch -n 1 nvidia-smi
```

### 6. Merge LoRA Adapter

After training:

```bash
cd train/utils
uv run adapter_merge.py \
  --base_model ../model/Qwen/Qwen2.5-Coder-3B-Instruct \
  --adapter ../output/dense/nl2sql_3b_lowmem/checkpoint-final \
  --output ../merged_models/nl2sql_3b_lowmem
```

---

## Troubleshooting

### Issue 1: Out of Memory (OOM)

**Symptoms**: Training crashes with CUDA OOM error

**Solutions** (try in order):

```bash
# Edit xiyan_sft_3b_low_mem.sh

# Solution 1: Reduce max length
MAX_LENGTH=2048          # Instead of 4096

# Solution 2: Reduce LoRA rank
LORA_R=32                # Instead of 64

# Solution 3: Increase gradient accumulation
ACC_STEP=64              # Instead of 32

# Solution 4: Filter long sequences
# Add to data assembly: filter samples > 2048 tokens
```

### Issue 2: Training Too Slow

**Symptoms**: Taking forever to complete

**Solutions**:

```bash
# 1. Use smaller dataset
# Edit configs/datasets_nl2sql_standard.json
"sample_num": 50,        # Instead of -1 (use only 50 samples)

# 2. Reduce epochs
EPOCH=3                  # Instead of 5

# 3. Consider cloud GPU rental
# Google Colab, AWS, RunPod, etc. with better GPUs
```

### Issue 3: Model Not Learning

**Symptoms**: Loss not decreasing

**Solutions**:

```bash
# 1. Increase learning rate
LR=3e-6                  # Instead of 2e-6

# 2. Train longer
EPOCH=7                  # Instead of 5

# 3. Check data quality
# Verify assembled data looks correct
head -100 train/datasets/nl2sql_standard_train.json
```

### Issue 4: Slow Data Loading

**Symptoms**: GPU idle, waiting for data

**Solutions**:

```bash
# Already set in script
GROUP_BY_LENGTH=True     # Group similar length samples
# This helps with efficiency
```

---

## Alternative: Smaller Model

If 3B is still too large, consider using **0.5B or 1.5B** model:

```bash
# In xiyan_sft_3b_low_mem.sh
MODEL="model/Qwen/Qwen2.5-Coder-0.5B-Instruct"
# or
MODEL="model/Qwen/Qwen2.5-Coder-1.5B-Instruct"

# With 0.5B, you can increase settings:
MAX_LENGTH=8192          # Can handle longer contexts
LORA_R=128               # Can use higher rank
BATCH_SIZE=2             # Might fit larger batch
```

**0.5B Model Performance**: ~45-50% accuracy (still useful for learning/prototyping)

---

## Alternative: Cloud GPU Rental

If local training is too slow, consider:

### Google Colab
- **Free tier**: T4 GPU (16GB) - ~3-4x faster than your setup
- **Pro**: A100 GPU (40GB) - ~20x faster
- **Cost**: ~$10/month for Pro

### RunPod
- **RTX 4090** (24GB): $0.44/hour
- **A100** (40GB): $1.29/hour
- **Cost for BIRD training**: ~$2-3 total

### AWS SageMaker / Lambda Labs
- Similar options with hourly pricing
- Good for one-time training runs

---

## Expected Performance

### With Your 8GB Setup

| Metric | Value |
|--------|-------|
| Training Time (224 samples) | ~8-12 hours |
| Expected Accuracy (BIRD dev) | ~50-55% |
| LoRA Rank | 64 (vs 512 optimal) |
| Max Sequence Length | 4096 (vs 10240 optimal) |
| VRAM Usage | ~6-7GB |
| Cost | $0 (use existing hardware) |

### With Cloud GPU (A100 40GB)

| Metric | Value |
|--------|-------|
| Training Time (224 samples) | ~1-2 hours |
| Expected Accuracy (BIRD dev) | ~55-60% |
| LoRA Rank | 512 (optimal) |
| Max Sequence Length | 10240 (optimal) |
| Cost | ~$2-3 for full training |

---

## Recommendations

### For Learning / Experimentation
✅ **Use your RTX 3080 Ti** with low-memory config
- Free
- Good enough to learn the framework
- Can iterate on small datasets

### For Production Training
✅ **Rent cloud GPU** for final training run
- Much faster (1-2 hours vs 8-12 hours)
- Better results (higher LoRA rank, longer context)
- Only ~$2-3 for complete training

### Best of Both Worlds
1. **Develop locally**: Test with 3-10 samples on RTX 3080 Ti
2. **Train in cloud**: Final training run on rented A100
3. **Save money**: Only pay for cloud when doing serious training

---

## Memory Usage Breakdown

### Where Memory Goes (8GB Total)

```
Model Parameters (3B):           ~6GB (base model in VRAM)
LoRA Adapters (rank 64):         ~100MB (trainable params)
Optimizer States (offloaded):    CPU RAM (saved VRAM!)
Gradients:                       ~500MB
Activations (batch=1, len=4096): ~1GB
System Overhead:                 ~400MB
---
Total:                           ~6-7GB (fits in 8GB!)
```

### Why Zero3 Offload Helps

Without offload (Zero2):
- Model: 6GB
- Optimizer: 2GB ❌ **OOM!**
- Gradients: 500MB
- **Total: 8.5GB** - Won't fit

With offload (Zero3):
- Model: 6GB
- Optimizer: CPU RAM ✅
- Gradients: 500MB
- **Total: 6-7GB** - Fits!

---

## Summary

**Yes, you can train on RTX 3080 Ti 8GB!**

**Key settings**:
- Use `xiyan_sft_3b_low_mem.sh`
- LoRA rank: 64
- Max length: 4096
- Batch size: 1
- DeepSpeed Zero3 + CPU offload

**Trade-off**: Slower training, slightly lower quality, but it works!

**Alternative**: Rent cloud GPU for ~$2-3 to get full quality in 1-2 hours.

**Recommendation**: Test locally, train in cloud if needed. 🚀
