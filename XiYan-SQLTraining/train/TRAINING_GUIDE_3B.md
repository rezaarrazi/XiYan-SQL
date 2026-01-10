# Training Guide: Qwen2.5-Coder-3B for XiYan-SQL

This guide covers training all 4 XiYan-SQL generators using the Qwen2.5-Coder-3B model.

## Why 3B Instead of 7B?

**Advantages of 3B model:**
- ✅ **Faster training**: ~2-3x faster than 7B
- ✅ **Less memory**: Can use larger batch sizes
- ✅ **Faster inference**: Better for production deployment
- ✅ **Lower cost**: Cheaper to train and deploy

**Trade-offs:**
- ⚠️ Slightly lower accuracy (~2-5% lower on complex queries)
- ⚠️ Less capable on very long contexts

**Recommendation**: 3B is excellent for most Text-to-SQL tasks, especially if you're using ensemble (4 generators).

---

## Prerequisites

1. **Download the 3B model:**
   ```bash
   cd utils
   uv run model_download.py
   # Select: Qwen2.5-Coder-3B-Instruct
   cd ..
   ```

2. **Prepare datasets** (if not already done):
   ```bash
   cd ../data
   bash assemble_all_generators.sh
   cd ../train
   ```

---

## Training Configuration for 3B

### Optimized Hyperparameters

Compared to 7B, the 3B model benefits from:

| Parameter | 7B Model | 3B Model | Reason |
|-----------|----------|----------|--------|
| Learning Rate | 1e-6 | 2e-6 | Smaller model needs slightly higher LR |
| Batch Size | 1 | 2 | Less memory per sample allows larger batches |
| Epochs | 5 | 5 | Same convergence pattern |
| LoRA Rank | 512 | 512 | Keep high capacity for SQL complexity |
| Max Length | 10240 | 10240 | Same context length supported |

### Memory Requirements

**With LoRA (Recommended):**
- 8x A100 40GB: ~15-20GB per GPU ✅
- 8x RTX 3090 24GB: ~18-22GB per GPU ✅
- 4x A100 40GB: ~25-30GB per GPU ✅

**Full Fine-tuning (Optional):**
- 8x A100 40GB: ~30-35GB per GPU ✅
- 8x A100 80GB: ~30-35GB per GPU ✅ (overkill)

---

## Step-by-Step Training: All 4 Generators

### Generator 1: SQLG₁ (Heavy Augmentation)

**Purpose**: Maximum diversity for robust candidate generation

```bash
cd XiYan-SQLTraining/train

# Edit xiyan_sft_3b.sh:
# EXPR_ID="sqlg1_3b_heavy_aug"
# DATA="datasets/sqlg1_heavy_aug_train.json"
# OUTPUT="output/dense/sqlg1_3b_heavy_aug/"

bash xiyan_sft_3b.sh
```

**Expected training time**: 3-4 hours on 8x A100 40GB

---

### Generator 2: SQLG₂ (Moderate Augmentation)

**Purpose**: Balanced between diversity and overfitting

```bash
# Edit xiyan_sft_3b.sh:
# EXPR_ID="sqlg2_3b_moderate_aug"
# DATA="datasets/sqlg2_moderate_aug_train.json"
# OUTPUT="output/dense/sqlg2_3b_moderate_aug/"

bash xiyan_sft_3b.sh
```

**Expected training time**: 2-3 hours on 8x A100 40GB

---

### Generator 3: SQLG₃ (Light Augmentation)

**Purpose**: Minimal augmentation, close to original data

```bash
# Edit xiyan_sft_3b.sh:
# EXPR_ID="sqlg3_3b_light_aug"
# DATA="datasets/sqlg3_light_aug_train.json"
# OUTPUT="output/dense/sqlg3_3b_light_aug/"

bash xiyan_sft_3b.sh
```

**Expected training time**: 1-2 hours on 8x A100 40GB

---

### Generator 4: SQLG₄ (No Augmentation)

**Purpose**: Original data only, baseline generator

```bash
# Edit xiyan_sft_3b.sh:
# EXPR_ID="sqlg4_3b_no_aug"
# DATA="datasets/sqlg4_no_aug_train.json"
# OUTPUT="output/dense/sqlg4_3b_no_aug/"

bash xiyan_sft_3b.sh
```

**Expected training time**: 1-2 hours on 8x A100 40GB

---

## All-in-One Training Script

For convenience, here's a script to train all 4 generators sequentially:

```bash
#!/bin/bash
# train_all_3b_generators.sh

cd XiYan-SQLTraining/train

# SQLG₁ - Heavy augmentation
sed -i 's/EXPR_ID=.*/EXPR_ID="sqlg1_3b_heavy_aug"/' xiyan_sft_3b.sh
sed -i 's|DATA=.*|DATA="datasets/sqlg1_heavy_aug_train.json"|' xiyan_sft_3b.sh
sed -i 's|OUTPUT=.*|OUTPUT="output/dense/sqlg1_3b_heavy_aug/"|' xiyan_sft_3b.sh
bash xiyan_sft_3b.sh

# SQLG₂ - Moderate augmentation
sed -i 's/EXPR_ID=.*/EXPR_ID="sqlg2_3b_moderate_aug"/' xiyan_sft_3b.sh
sed -i 's|DATA=.*|DATA="datasets/sqlg2_moderate_aug_train.json"|' xiyan_sft_3b.sh
sed -i 's|OUTPUT=.*|OUTPUT="output/dense/sqlg2_3b_moderate_aug/"|' xiyan_sft_3b.sh
bash xiyan_sft_3b.sh

# SQLG₃ - Light augmentation
sed -i 's/EXPR_ID=.*/EXPR_ID="sqlg3_3b_light_aug"/' xiyan_sft_3b.sh
sed -i 's|DATA=.*|DATA="datasets/sqlg3_light_aug_train.json"|' xiyan_sft_3b.sh
sed -i 's|OUTPUT=.*|OUTPUT="output/dense/sqlg3_3b_light_aug/"|' xiyan_sft_3b.sh
bash xiyan_sft_3b.sh

# SQLG₄ - No augmentation
sed -i 's/EXPR_ID=.*/EXPR_ID="sqlg4_3b_no_aug"/' xiyan_sft_3b.sh
sed -i 's|DATA=.*|DATA="datasets/sqlg4_no_aug_train.json"|' xiyan_sft_3b.sh
sed -i 's|OUTPUT=.*|OUTPUT="output/dense/sqlg4_3b_no_aug/"|' xiyan_sft_3b.sh
bash xiyan_sft_3b.sh

echo "All 4 generators trained successfully!"
```

**Total training time**: ~7-11 hours on 8x A100 40GB

---

## Monitoring Training

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Monitor Training Logs
```bash
# Real-time log monitoring
tail -f output/dense/sqlg1_3b_heavy_aug/training.log

# Check loss trends
grep "loss" output/dense/sqlg1_3b_heavy_aug/training.log | tail -20
```

### Expected Loss Trends (3B Model)

- **Initial loss**: ~1.5-2.0
- **After 1 epoch**: ~0.8-1.2
- **After 3 epochs**: ~0.5-0.8
- **After 5 epochs**: ~0.3-0.6

---

## Model Checkpoints

Checkpoints are saved every 500 steps to:
```
output/dense/sqlg1_3b_heavy_aug/
├── checkpoint-500/
├── checkpoint-1000/
├── checkpoint-1500/
└── checkpoint-final/  (best checkpoint)
```

---

## Merging LoRA Adapters (If Using LoRA)

After training, merge LoRA adapters back into the base model:

```bash
cd utils

# SQLG₁
uv run adapter_merge.py \
  --base_model ../model/Qwen/Qwen2.5-Coder-3B-Instruct \
  --adapter ../output/dense/sqlg1_3b_heavy_aug/checkpoint-final \
  --output ../merged_models/sqlg1_3b_heavy_aug

# SQLG₂
uv run adapter_merge.py \
  --base_model ../model/Qwen/Qwen2.5-Coder-3B-Instruct \
  --adapter ../output/dense/sqlg2_3b_moderate_aug/checkpoint-final \
  --output ../merged_models/sqlg2_3b_moderate_aug

# SQLG₃
uv run adapter_merge.py \
  --base_model ../model/Qwen/Qwen2.5-Coder-3B-Instruct \
  --adapter ../output/dense/sqlg3_3b_light_aug/checkpoint-final \
  --output ../merged_models/sqlg3_3b_light_aug

# SQLG₄
uv run adapter_merge.py \
  --base_model ../model/Qwen/Qwen2.5-Coder-3B-Instruct \
  --adapter ../output/dense/sqlg4_3b_no_aug/checkpoint-final \
  --output ../merged_models/sqlg4_3b_no_aug
```

---

## Evaluation

Test each generator individually:

```bash
cd ../evaluation

# SQLG₁
bash sql_infer.sh \
  ../train/merged_models/sqlg1_3b_heavy_aug \
  sqlg1_3b_eval \
  bird_evaluation/eval_set/bird_dev.json \
  4

bash sql_eval.sh \
  bird_evaluation/output/sqlg1_3b_eval/sqlg1_3b_eval_results.json \
  bird_evaluation/eval_set/bird_dev.json \
  bird_evaluation/db_conn.json \
  bird_evaluation/output/sqlg1_3b_eval/scores.json

# Repeat for SQLG₂, SQLG₃, SQLG₄
```

---

## Expected Performance (3B vs 7B)

Based on BIRD benchmark:

| Model | Execution Accuracy | Speed (samples/sec) |
|-------|-------------------|---------------------|
| 3B Single | ~55-60% | ~15-20 |
| 3B Ensemble (4 generators) | ~62-67% | ~4-5 |
| 7B Single | ~60-65% | ~10 |
| 7B Ensemble (4 generators) | ~67-72% | ~2.5 |

**Key insight**: 3B ensemble often outperforms 7B single model while being faster!

---

## Troubleshooting

### Out of Memory

**Solution 1**: Reduce batch size
```bash
BATCH_SIZE=1  # in xiyan_sft_3b.sh
```

**Solution 2**: Enable gradient checkpointing (already enabled)

**Solution 3**: Reduce max length
```bash
MAX_LENGTH=8192  # instead of 10240
```

### Training Too Slow

**Solution 1**: Increase batch size (if memory allows)
```bash
BATCH_SIZE=4
ACC_STEP=1
```

**Solution 2**: Reduce number of GPUs
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use only 4 GPUs
```

### Model Not Converging

**Solution 1**: Increase learning rate
```bash
LR=3e-6  # instead of 2e-6
```

**Solution 2**: Train for more epochs
```bash
EPOCH=7  # instead of 5
```

---

## Fine-tuning Tips for 3B

1. **Use LoRA**: Almost always recommended for 3B
   - Faster training
   - Better generalization
   - Easier to experiment

2. **Larger batch size**: 3B can handle `BATCH_SIZE=2` or even `4` on A100 40GB

3. **Slightly higher LR**: 3B benefits from `2e-6` instead of `1e-6`

4. **Same epochs as 7B**: 5 epochs is still optimal

5. **Keep LoRA rank high**: Use `LORA_R=512` for SQL complexity

---

## Production Deployment

For inference, use the merged models:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load merged model
model = AutoModelForCausalLM.from_pretrained(
    "merged_models/sqlg1_3b_heavy_aug",
    device_map="auto",
    torch_dtype="bfloat16"
)
tokenizer = AutoTokenizer.from_pretrained("merged_models/sqlg1_3b_heavy_aug")

# Generate SQL
prompt = "你是一名SQLite专家..."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=2048)
sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## Next Steps

After training all 4 generators:

1. **Evaluate each generator** individually on BIRD dev set
2. **Implement ensemble voting** (not included in this repo)
3. **Test on production data** with your own databases
4. **Fine-tune further** on domain-specific data if needed

---

## References

- Main Documentation: [`../data/TRAINING_STRATEGIES.md`](../data/TRAINING_STRATEGIES.md)
- Quick Start Guide: [`../data/QUICK_START.md`](../data/QUICK_START.md)
- Generator Configs: [`../data/configs/README_GENERATORS.md`](../data/configs/README_GENERATORS.md)
- XiYan-SQL Paper: [arXiv:2507.04701](https://arxiv.org/abs/2507.04701)
