# XiYan-SQL Training: Quick Start Guide

## 🚀 5-Minute Overview

This guide gets you started with training a single SQL model. For comprehensive details, see [TRAINING_STRATEGIES.md](TRAINING_STRATEGIES.md).

---

## Prerequisites

- **Linux or WSL2** (DeepSpeed requirement)
- **CUDA 12.6+** (PyTorch 2.9.0)
- **Python 3.10+**
- **GPU**: At least 1x A100 (40GB) or equivalent

---

## Quick Setup

```bash
# 1. Clone and enter directory
cd XiYan-SQLTraining

# 2. Install dependencies (using uv - recommended)
uv sync

# No activation needed - all scripts use `uv run`
```

---

## Training Your First Model (3 Steps)

### Step 1: Prepare Data

**Option A: Use Test Data** (already included):
```bash
cd data
# Test data is already processed at:
# data_warehouse/test_data/processed_data/test_train_nl2sqlite.json
```

**Option B: Process Your Own Data**:
```bash
# If you have raw BIRD data:
cd data
bash data_processing.sh \
  data_warehouse/bird_train/raw_data/train.json \
  data_warehouse/bird_train/db_conn.json \
  data_warehouse/bird_train/processed_data/ \
  data_warehouse/bird_train/mschema/ \
  configs/datasets_all.json
```

### Step 2: Assemble Training Dataset

```bash
cd data
bash data_assembler.sh \
  configs/datasets_example.json \
  ../train/datasets/my_first_train.json
```

### Step 3: Train Model

```bash
cd ../train

# Download base model first
cd utils
uv run model_download.py
# Select: Qwen2.5-Coder-7B-Instruct

# Edit xiyan_sft.sh - change these lines:
# EXPR_ID="my_first_model"
# DATA="datasets/my_first_train.json"
# OUTPUT="output/dense/my_first_model/"

cd ..
bash xiyan_sft.sh
```

**Training time**: ~4-6 hours on 8x A100 GPUs

---

## Inference & Evaluation

```bash
cd evaluation

# 1. Run inference
bash sql_infer.sh \
  ../train/output/dense/my_first_model/checkpoint-final \
  my_eval \
  bird_evaluation/eval_set/bird_dev.json \
  4

# 2. Evaluate results
bash sql_eval.sh \
  bird_evaluation/output/my_eval/my_eval_results.json \
  bird_evaluation/eval_set/bird_dev.json \
  bird_evaluation/db_conn.json \
  bird_evaluation/output/my_eval/scores.json
```

---

## Key Configuration Files

| File | Purpose |
|------|---------|
| `data/configs/datasets_*.json` | Define which datasets to use |
| `train/xiyan_sft.sh` | Standard SFT training config |
| `train/xiyan_momq_sft.sh` | MOE multi-dialect training |
| `train/config/*.yaml` | DeepSpeed configurations |

---

## Common Issues & Solutions

### 1. Out of Memory Error

**Solution**: Enable LoRA or reduce batch size
```bash
# In xiyan_sft.sh:
USE_LORA=True
LORA_R=512
BATCH_SIZE=1
ACC_STEP=4  # Increase accumulation steps
```

### 2. Import Errors (llama-index)

Already fixed in the codebase! The code now has fallback imports for older llama-index versions.

### 3. Download Too Slow

**Solution**: Download BIRD data separately with a download manager, then place in:
```
XiYan-SQLTraining/data/data_warehouse/bird_train/
```

---

## Next Steps

After training your first model:

1. **Experiment with augmentation** - See [TRAINING_STRATEGIES.md § Data Augmentation](TRAINING_STRATEGIES.md#data-augmentation-pipeline)
2. **Try multi-dialect training** - See [TRAINING_STRATEGIES.md § MOMQ MOE](TRAINING_STRATEGIES.md#momq-moe-architecture)
3. **Create self-refine data** - See [TRAINING_STRATEGIES.md § Task 3: Self-Refinement](TRAINING_STRATEGIES.md#task-3-self-refinement)
4. **Train 4 generators** - See [TRAINING_STRATEGIES.md § Creating the Four Generators](TRAINING_STRATEGIES.md#creating-the-four-generators)

---

## Useful Commands Reference

```bash
# Check GPU usage
nvidia-smi

# Monitor training logs
tail -f train/output/dense/my_first_model/training.log

# Find latest checkpoint
ls -lt train/output/dense/my_first_model/ | head

# Merge LoRA adapter
cd train
uv run utils/adapter_merge.py \
  --base_model model/Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter output/dense/my_first_model/checkpoint-XXX \
  --output merged_models/my_model

# Run inference with merged model
cd evaluation
bash sql_infer.sh ../train/merged_models/my_model eval_v1 eval_set.json 4
```

---

## XiYan-SQL System Pipeline

**This repository trains generators for Stage 3 of the full XiYan-SQL system:**

```
┌─────────────────────────────────────────────────────┐
│               Full XiYan-SQL Pipeline               │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. Inputs → Question, Evidence, Full DB Schema    │
│               ↓                                     │
│  2. Schema Filter → Filtered Schema Subset         │
│               ↓                                     │
│  3. Multiple SQL Generation → 4 SQL Candidates     │
│      (SQLG₁, SQLG₂, SQLG₃, SQLG₄)                 │
│               ↓                                     │
│  4. SQL Selection → Final SQL Output               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**What This Repo Does**: Trains the 4 generators (SQLG₁-₄) in Stage 3
**What This Repo Doesn't Do**: Schema filtering at inference time, ensemble voting/selection

---

## Architecture Quick Reference

```
XiYan-SQLTraining/
├── data/                    # Data processing
│   ├── data_processing.py   # Process raw data → M-Schema
│   ├── data_assembler.py    # Assemble training dataset
│   ├── data_utils/          # Utilities
│   │   ├── m_schema.py      # M-Schema implementation
│   │   ├── schema_engine.py # Extract schema from DB
│   │   ├── prompt_utils.py  # Task-specific prompts
│   │   └── aug_ops/         # Data augmentation
│   └── configs/             # Dataset configurations
│
├── train/                   # Model training
│   ├── sft4xiyan.py         # Main training script
│   ├── xiyan_sft.sh         # Standard SFT config
│   ├── xiyan_momq_sft.sh    # MOE training config
│   ├── trainer/             # Custom trainers
│   ├── model/               # Custom model implementations
│   └── config/              # DeepSpeed configs
│
└── evaluation/              # Inference & evaluation
    ├── sql_infer.py         # Generate SQL predictions
    ├── sql_eval.py          # Evaluate accuracy
    └── eval_utils/          # Evaluation utilities
```

---

## Training Parameters Quick Guide

### Standard SFT (7B Model)
```bash
MODEL="Qwen2.5-Coder-7B-Instruct"
EPOCH=5
LR=1e-6
BATCH_SIZE=1
ACC_STEP=2
MAX_LENGTH=10240
USE_LORA=True
LORA_R=512
```

### MOE Training (Multi-Dialect)
```bash
MODEL="Qwen2.5-Coder-7B-Instruct"
USE_MOE_LORA=True
num_experts=24
num_experts_per_tok=2
enable_dialect_router=True
LR=1e-5
MAX_LENGTH=8192
```

### Large Model (32B Full FT)
```bash
MODEL="Qwen2.5-Coder-32B-Instruct"
EPOCH=3
LR=5e-7
USE_LORA=False
# Requires 8x A100 80GB
```

---

## Data Format Quick Reference

### Input to Data Processing
```json
{
  "db_id": "movie_platform",
  "question": "What are the top rated movies?",
  "evidence": "top rated means highest rating_score",
  "SQL": "SELECT movie_title FROM movies ORDER BY rating_score DESC LIMIT 10",
  "db_schema": "CREATE TABLE movies (...)"  // Optional if DB connection available
}
```

### Output from Data Assembly
```json
{
  "id": 0,
  "conversations": [
    {
      "role": "user",
      "content": "你是一名SQLite专家，现在需要阅读并理解下面的【数据库schema】描述..."
    },
    {
      "role": "assistant",
      "content": "SELECT movie_title FROM movies ORDER BY rating_score DESC LIMIT 10"
    }
  ],
  "sql_type": "sqlite"
}
```

---

## Performance Expectations

| Model Size | Training Time | Inference Speed | Accuracy (BIRD Dev) |
|------------|---------------|-----------------|---------------------|
| 7B (LoRA) | 4-6 hours | ~10 samples/sec | ~60-65% |
| 14B (LoRA) | 8-10 hours | ~5 samples/sec | ~65-70% |
| 7B MOE | 10-12 hours | ~8 samples/sec | ~62-67% |
| 32B (Full) | 18-24 hours | ~2 samples/sec | ~70-75% |

*Based on 8x A100 40GB GPUs*

---

## Support & Resources

- **Detailed Guide**: [TRAINING_STRATEGIES.md](TRAINING_STRATEGIES.md) (53KB comprehensive documentation)
- **Repository**: [GitHub - XiYan-SQL](https://github.com/alibaba/XiYan-SQL)
- **Paper**: [arXiv:2507.04701](https://arxiv.org/abs/2507.04701)
- **Issues**: [GitHub Issues](https://github.com/alibaba/XiYan-SQL/issues)

---

**Happy Training!** 🎉

For advanced features like:
- Multi-task training (self-refine, candidate selection)
- Custom augmentation strategies
- Multi-generator ensemble
- Production deployment

→ See the complete [TRAINING_STRATEGIES.md](TRAINING_STRATEGIES.md)
