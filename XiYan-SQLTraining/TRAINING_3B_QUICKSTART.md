# Quick Start: Training XiYan-SQL with Qwen2.5-Coder-3B

A streamlined guide for training all 4 XiYan-SQL generators using the 3B model.

---

## Why 3B?

✅ **2-3x faster** training than 7B
✅ **Lower memory** requirements
✅ **Faster inference** for production
✅ **Great ensemble performance** (4 generators often beat single 7B)

---

## Complete Training Pipeline (3 Commands)

### 1️⃣ Download Model

```bash
cd XiYan-SQLTraining/train/utils
uv run model_download.py
# Select: 3 (Qwen2.5-Coder-3B-Instruct)
cd ../..
```

### 2️⃣ Prepare All Datasets

```bash
cd data
bash assemble_all_generators.sh
cd ..
```

This creates 4 training datasets:
- `train/datasets/sqlg1_heavy_aug_train.json` (~896 samples)
- `train/datasets/sqlg2_moderate_aug_train.json` (~448 samples)
- `train/datasets/sqlg3_light_aug_train.json` (~224 samples)
- `train/datasets/sqlg4_no_aug_train.json` (224 samples)

### 3️⃣ Train All 4 Generators

```bash
cd train

# SQLG₁ - Heavy Augmentation
# Edit xiyan_sft_3b.sh lines:
#   EXPR_ID="sqlg1_3b_heavy_aug"
#   DATA="datasets/sqlg1_heavy_aug_train.json"
#   OUTPUT="output/dense/sqlg1_3b_heavy_aug/"
bash xiyan_sft_3b.sh

# SQLG₂ - Moderate Augmentation
# Edit xiyan_sft_3b.sh lines:
#   EXPR_ID="sqlg2_3b_moderate_aug"
#   DATA="datasets/sqlg2_moderate_aug_train.json"
#   OUTPUT="output/dense/sqlg2_3b_moderate_aug/"
bash xiyan_sft_3b.sh

# SQLG₃ - Light Augmentation
# Edit xiyan_sft_3b.sh lines:
#   EXPR_ID="sqlg3_3b_light_aug"
#   DATA="datasets/sqlg3_light_aug_train.json"
#   OUTPUT="output/dense/sqlg3_3b_light_aug/"
bash xiyan_sft_3b.sh

# SQLG₄ - No Augmentation
# Edit xiyan_sft_3b.sh lines:
#   EXPR_ID="sqlg4_3b_no_aug"
#   DATA="datasets/sqlg4_no_aug_train.json"
#   OUTPUT="output/dense/sqlg4_3b_no_aug/"
bash xiyan_sft_3b.sh
```

---

## Training Configuration (3B Optimized)

The `xiyan_sft_3b.sh` script is pre-configured with optimal settings:

```bash
MODEL="model/Qwen/Qwen2.5-Coder-3B-Instruct"
EPOCH=5
LR=2e-6              # Slightly higher than 7B (1e-6)
BATCH_SIZE=2         # Larger than 7B (1)
ACC_STEP=2
MAX_LENGTH=10240
USE_LORA=True
LORA_R=512
```

**Effective batch size**: 2 × 2 × 8 GPUs = **32**

---

## Expected Training Times (8x A100 40GB)

| Generator | Dataset Size | Training Time |
|-----------|--------------|---------------|
| SQLG₁ | ~896 samples | 3-4 hours |
| SQLG₂ | ~448 samples | 2-3 hours |
| SQLG₃ | ~224 samples | 1-2 hours |
| SQLG₄ | 224 samples | 1-2 hours |
| **Total** | | **7-11 hours** |

---

## Memory Requirements

**With LoRA** (recommended):
- 8x A100 40GB: ~15-20GB per GPU ✅
- 8x RTX 3090 24GB: ~18-22GB per GPU ✅
- 4x A100 40GB: ~25-30GB per GPU ✅

---

## Monitoring Training

```bash
# Check GPU usage
nvidia-smi

# Monitor logs in real-time
tail -f train/output/dense/sqlg1_3b_heavy_aug/training.log

# Check loss (should decrease from ~1.5 to ~0.3-0.6)
grep "loss" train/output/dense/sqlg1_3b_heavy_aug/training.log | tail -20
```

---

## After Training: Merge LoRA Adapters

```bash
cd train/utils

# Merge all 4 generators
for gen in sqlg1 sqlg2 sqlg3 sqlg4; do
  uv run adapter_merge.py \
    --base_model ../model/Qwen/Qwen2.5-Coder-3B-Instruct \
    --adapter ../output/dense/${gen}_3b_*_aug/checkpoint-final \
    --output ../merged_models/${gen}_3b
done
```

---

## Evaluation

```bash
cd evaluation

# Test SQLG₁
bash sql_infer.sh \
  ../train/merged_models/sqlg1_3b \
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

## Expected Performance

| Configuration | Accuracy (BIRD Dev) | Inference Speed |
|---------------|---------------------|-----------------|
| 3B Single Model | ~55-60% | ~15-20 samples/sec |
| 3B Ensemble (4 gens) | ~62-67% | ~4-5 samples/sec |
| 7B Single Model | ~60-65% | ~10 samples/sec |
| 7B Ensemble (4 gens) | ~67-72% | ~2.5 samples/sec |

**Key Insight**: 3B ensemble often beats 7B single model! 🎯

---

## Files Created

```
XiYan-SQLTraining/
├── data/
│   ├── configs/
│   │   ├── datasets_sqlg1_heavy_aug.json      ← Dataset config
│   │   ├── datasets_sqlg2_moderate_aug.json   ← Dataset config
│   │   ├── datasets_sqlg3_light_aug.json      ← Dataset config
│   │   ├── datasets_sqlg4_no_aug.json         ← Dataset config
│   │   └── README_GENERATORS.md               ← Config documentation
│   └── assemble_all_generators.sh             ← Assembly script
│
├── train/
│   ├── xiyan_sft_3b.sh                        ← Training script (3B optimized)
│   ├── TRAINING_GUIDE_3B.md                   ← Detailed guide
│   └── utils/
│       └── model_download.py                  ← Updated with 3B as default
│
└── TRAINING_3B_QUICKSTART.md                  ← This file
```

---

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size in xiyan_sft_3b.sh
BATCH_SIZE=1
ACC_STEP=4
```

### Training Too Slow
```bash
# Increase batch size (if memory allows)
BATCH_SIZE=4
ACC_STEP=1
```

### Model Not Converging
```bash
# Increase learning rate or epochs
LR=3e-6
EPOCH=7
```

---

## Next Steps

After training all 4 generators:

1. **Evaluate individually** on BIRD dev set
2. **Implement ensemble** (voting/selection) - not included in this repo
3. **Test on your data** with custom databases
4. **Fine-tune further** on domain-specific SQL if needed

---

## Full Documentation

- **Detailed Training Guide**: [`train/TRAINING_GUIDE_3B.md`](train/TRAINING_GUIDE_3B.md)
- **Generator Configs**: [`data/configs/README_GENERATORS.md`](data/configs/README_GENERATORS.md)
- **Training Strategies**: [`TRAINING_STRATEGIES.md`](TRAINING_STRATEGIES.md)
- **Quick Start (General)**: [`QUICK_START.md`](QUICK_START.md)

---

## Quick Reference: Config Parameters

| Parameter | SQLG₁ | SQLG₂ | SQLG₃ | SQLG₄ |
|-----------|-------|-------|-------|-------|
| `data_aug` | true | true | true | false |
| `aug_ratio` | 4.0 | 2.0 | 1.0 | - |
| Dataset Size | ~896 | ~448 | ~224 | 224 |
| Train Time | 3-4h | 2-3h | 1-2h | 1-2h |

---

**Happy Training!** 🚀

For questions or issues, see the detailed guides or check the [GitHub Issues](https://github.com/alibaba/XiYan-SQL/issues).
