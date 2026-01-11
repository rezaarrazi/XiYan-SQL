# XiYan-SQLTraining Framework

## News 🔥
+ `2025-10-30` 🌟 We are pleased to announce the release of the first version of the XiYan-SQL training framework **XiYan-SQLTraining**. We welcome everyone to use it, and we will be adding more information to enhance this framework in the future.

---

## Table of Contents
1. [Introduction](#introduction)
2. [What This Repository Trains](#what-this-repository-trains)
3. [Quick Start](#quick-start)
4. [System Requirements](#system-requirements)
5. [Training Guide](#training-guide)
6. [Model Evaluation](#model-evaluation)
7. [Multi-Task Training](#multi-task-training)
8. [Architecture](#architecture)
9. [Citation](#citation)

---

## Introduction

The XiYan-SQLTraining framework is a post-training framework specifically designed for the Text-to-SQL task developed by XiYan. Currently, it mainly supports the following capabilities:

- [x] Conversion of raw data to training data
- [x] Training data augmentation
- [x] Fine-tuning basic models for Text2SQL tasks
- [x] Training the XiYanSQL MOE multi-dialect model
- [x] Model inference/evaluation
- [ ] Continued GRPO training for Text2SQL
- [ ] Integration of different types of SQL models

The framework is continuously being improved, and we welcome contributions from users!

---

## What This Repository Trains

This repository trains **SQLG₁** - a Text-to-SQL model for **SQLite dialect**.

### Generator Types (SQL Formatting Variations)

The XiYan-SQL paper describes 4 generators with different SQL formatting styles:

- ✅ **SQLG₁**: Standard NL2SQL (no special formatting) - **Fully implemented**
- ❌ **SQLG₂**: Complex writing patterns (chunked SQL, CTEs) - Not yet implemented
- ❌ **SQLG₃**: Standardized presentation styles - Not yet implemented
- ❌ **SQLG₄**: Comprehensive mix - Not yet implemented

### Multi-Task Training (Paper's 4 Tasks)

The paper trains SQLG₁ on 4 different tasks:

1. ✅ **Text-to-SQL** (Question → SQL) - **Fully implemented** ← **Default training**
2. ❌ **Question Inference** (SQL → Question) - Not implemented
3. ❌ **Evidence Inference** (Question+SQL → Evidence) - Not implemented
4. ⚠️ **Self-Refine** (Wrong SQL → Corrected SQL) - Template exists, needs data

**Current default**: Trains **Task 1 only** (Text-to-SQL) on **SQLG₁** format with **SQLite** dialect.

This gives you a solid baseline Text-to-SQL model. While the paper uses all 4 tasks for better performance, Task 1 alone is sufficient for good results (~55-60% accuracy on BIRD benchmark).

---

## Quick Start

### Prerequisites

- **Linux or WSL2** (DeepSpeed requires Linux-specific libraries)
- **CUDA 12.6+** (PyTorch 2.9.0 requirement)
- **Python 3.10+**
- **GPU Options**:
  - **Recommended**: 8x A100 40GB GPUs (or 4x with adjusted batch size)
  - **Low Memory**: 1x RTX 3080 Ti 8GB (see [Low Memory Training](LOW_MEMORY_TRAINING.md) - slower but works!)

### Complete Training Flow (3 Steps)

#### Step 1: Setup Environment

```bash
cd XiYan-SQLTraining
uv sync  # Installs all dependencies
```

All scripts use `uv run` automatically - no manual activation needed!

#### Step 2: Download Model

```bash
cd train/utils
uv run model_download.py
# Select: 3 (Qwen2.5-Coder-3B-Instruct)
cd ../..
```

#### Step 3: Prepare Data & Train

```bash
# Assemble training data
cd data
bash data_assembler.sh \
  configs/datasets_nl2sql_standard.json \
  ../train/datasets/nl2sql_standard_train.json

# Train the model
cd ../train
bash xiyan_sft_3b.sh

# Monitor training (in another terminal)
swanlab watch train/output/dense/nl2sql_3b_standard/swanlab
# Opens dashboard at http://localhost:5092

# Merge LoRA adapter (if using LoRA)
cd utils
uv run adapter_merge.py \
  --base_model ../model/Qwen/Qwen2.5-Coder-3B-Instruct \
  --adapter ../output/dense/nl2sql_3b_standard/checkpoint-final \
  --output ../merged_models/nl2sql_3b_standard
```

**Training time**: ~1-2 hours on 8x A100 40GB (with 224 BIRD samples)

---

## System Requirements

### Platform
- **Linux required** - DeepSpeed requires Linux-specific libraries (libaio) and has symlink permission issues on Windows
- **Windows users must use WSL2** (Windows Subsystem for Linux 2)

### CUDA
- **Minimum: CUDA 12.6** (PyTorch 2.9.0 is built for CUDA 12.6)
- Verify your CUDA version: `nvcc --version` or `nvidia-smi`

### Python
- Python 3.10 or higher

### GPU Memory (with LoRA)
- 8x A100 40GB: ~18-20GB per GPU ✅
- 8x RTX 3090 24GB: ~18-22GB per GPU ✅
- 4x A100 40GB: ~25-30GB per GPU ✅
- **1x RTX 3080 Ti 8GB**: ~6-7GB ✅ (see [Low Memory Training Guide](LOW_MEMORY_TRAINING.md))

---

## Training Guide

### Important: Experiment Tracking

This framework uses **SwanLab** for automatic experiment tracking. All training metrics (loss, learning rate, GPU usage) are automatically logged.

**Quick Access**:
- **Local Dashboard**: `swanlab watch train/output/dense/<experiment_name>/swanlab`
- **Cloud Dashboard**: `swanlab login` then visit https://swanlab.cn/@your-username/SQLTrainer

No configuration needed - it works out of the box! See [Monitoring Training](#monitor-training) for details.

---

### 1. Data Preparation

#### Option A: Use Test Data (Quick Start)

Test data is already processed and ready:
```bash
cd data
ls -l data_warehouse/test_data/processed_data/test_train_nl2sqlite.json
# 3 samples ready to use
```

#### Option B: Process BIRD Training Data

If you have BIRD dataset downloaded:

```bash
cd data

# Step 1: Process raw BIRD data
bash data_processing.sh \
  data_warehouse/bird_train/raw_data/train.json \
  data_warehouse/bird_train/db_conn.json \
  data_warehouse/bird_train/processed_data/ \
  data_warehouse/bird_train/mschema/ \
  configs/datasets_all.json

# Step 2: Assemble training dataset
bash data_assembler.sh \
  configs/datasets_nl2sql_standard.json \
  ../train/datasets/nl2sql_standard_train.json
```

**Expected output**: `train/datasets/nl2sql_standard_train.json` (224 samples with BIRD data)

#### Training Data Format

The assembled training data follows this structure:

```json
[
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
    "sql_type": "nl2sqlite"
  }
]
```

### 2. Model Training

#### Download Base Model

```bash
cd train/utils
uv run model_download.py
# Interactive menu - select option 3 for Qwen2.5-Coder-3B-Instruct
```

#### Train with Optimized Script

The training script `xiyan_sft_3b.sh` is pre-configured with optimal settings for 3B model:

```bash
cd train
bash xiyan_sft_3b.sh
```

**Default configuration**:
```bash
MODEL="model/Qwen/Qwen2.5-Coder-3B-Instruct"
EPOCH=5
LR=2e-6                  # Optimized for 3B
BATCH_SIZE=2             # Larger than 7B (1)
ACC_STEP=2
MAX_LENGTH=10240
USE_LORA=True
LORA_R=512
```

**Effective batch size**: 2 × 2 × 8 GPUs = **32**

#### Monitor Training

**SwanLab Dashboard** (Recommended - automatically enabled):

Training metrics are automatically tracked with SwanLab. View real-time progress:

```bash
# Option 1: Local web dashboard
swanlab watch train/output/dense/nl2sql_3b_standard/swanlab
# Opens browser at http://localhost:5092

# Option 2: Cloud dashboard (optional - requires login)
swanlab login  # First time only
# Then visit: https://swanlab.cn/@your-username/SQLTrainer
```

**Command Line Monitoring**:

```bash
# Check GPU usage
nvidia-smi

# Monitor training logs
tail -f train/output/dense/nl2sql_3b_standard/training.log

# Expected loss progression:
# Initial: ~1.5-2.0
# After 1 epoch: ~0.8-1.2
# After 3 epochs: ~0.5-0.8
# After 5 epochs: ~0.3-0.6
```

**What SwanLab Tracks**:
- Loss curves (training/validation)
- Learning rate schedule
- GPU memory usage
- Training speed (steps/sec)
- Custom metrics

### 3. Merge LoRA Adapter

After training completes (if using LoRA):

```bash
cd train/utils

uv run adapter_merge.py \
  --base_model ../model/Qwen/Qwen2.5-Coder-3B-Instruct \
  --adapter ../output/dense/nl2sql_3b_standard/checkpoint-final \
  --output ../merged_models/nl2sql_3b_standard
```

### Configuration File

Location: `data/configs/datasets_nl2sql_standard.json`

```json
{
  "bird_train_processed_data_train_nl2sqlite": {
    "data_path": "data_warehouse/bird_train/processed_data/train_nl2sqlite.json",
    "sample_num": -1,
    "sum_num": 221,
    "task_name": "nl2sqlite",
    "data_aug": false
  },
  "test_data_processed_data_test_train_nl2sqlite": {
    "data_path": "data_warehouse/test_data/processed_data/test_train_nl2sqlite.json",
    "sample_num": -1,
    "sum_num": 3,
    "task_name": "nl2sqlite",
    "data_aug": false
  }
}
```

**Key parameters**:
- `task_name: "nl2sqlite"` - Text-to-SQL task for SQLite
- `data_aug: false` - No data augmentation (clean training)
- `sample_num: -1` - Use all samples

---

## Model Evaluation

### Run Inference

```bash
cd evaluation

bash sql_infer.sh \
  ../train/merged_models/nl2sql_3b_standard \
  nl2sql_eval \
  bird_evaluation/eval_set/bird_dev.json \
  4
```

**Parameters**:
- `model_name_or_path`: Path to merged model
- `expr_version`: Experiment name
- `test_set_path`: Test dataset path
- `batch_size`: Concurrent processing size

### Evaluate Results

```bash
bash sql_eval.sh \
  bird_evaluation/output/nl2sql_eval/nl2sql_eval_results.json \
  bird_evaluation/eval_set/bird_dev.json \
  bird_evaluation/db_conn.json \
  bird_evaluation/output/nl2sql_eval/scores.json
```

**Parameters**:
- `pred_sql_path`: Predicted SQL path
- `test_sql_path`: Test set with ground-truth SQL
- `db_conn_config`: Database configuration
- `save_eval_path`: Path to save evaluation results

### Expected Performance

| Model | Execution Accuracy (BIRD Dev) | Inference Speed |
|-------|-------------------------------|-----------------|
| 3B Single (Task 1 only) | ~55-60% | ~15-20 samples/sec |
| 3B (Task 1 + Self-Refine) | ~58-63% | ~15-20 samples/sec |
| Paper's SQLG₁ (all 4 tasks) | ~60-65% | - |
| Paper's 4-generator ensemble | ~67-72% | - |

---

## Multi-Task Training

### Paper's Approach (4 Tasks)

The paper trains SQLG₁ on 4 different tasks to improve model robustness:

#### Task 1: Text-to-SQL ✅ Implemented
**Input**: Question + Schema + Evidence
**Output**: SQL query

```
Question: What are the top rated movies?
Schema: CREATE TABLE movies (movie_id, movie_title, rating_score)
Evidence: top rated means highest rating_score
→ Output: SELECT movie_title FROM movies ORDER BY rating_score DESC LIMIT 10
```

#### Task 2: Question Inference ❌ Not Implemented
**Input**: Schema + Evidence + SQL
**Output**: Natural language question

```
Schema: CREATE TABLE movies (...)
Evidence: top rated means highest rating_score
SQL: SELECT movie_title FROM movies ORDER BY rating_score DESC LIMIT 10
→ Output: What are the top rated movies?
```

#### Task 3: Evidence Inference ❌ Not Implemented
**Input**: Schema + Question + SQL
**Output**: Evidence/hints

```
Schema: CREATE TABLE movies (...)
Question: What are the top rated movies?
SQL: SELECT movie_title FROM movies ORDER BY rating_score DESC LIMIT 10
→ Output: top rated means highest rating_score
```

#### Task 4: Self-Refine ⚠️ Template Exists
**Input**: Question + Schema + Evidence + Wrong SQL + Error
**Output**: Corrected SQL

```
Question: What are the top rated movies?
Schema: CREATE TABLE movies (...)
Evidence: top rated means highest rating_score
Wrong SQL: SELECT * FROM movies WHERE rating_score > 8
Error: Results don't match expected output
→ Output: SELECT movie_title FROM movies ORDER BY rating_score DESC LIMIT 10
```

### Implementation Status

| Task | Template | Status | Config Name |
|------|----------|--------|-------------|
| Text-to-SQL | `NL2SQLITE_TEMPLATE` | ✅ Implemented | `nl2sqlite` |
| Question Inference | - | ❌ Not implemented | - |
| Evidence Inference | - | ❌ Not implemented | - |
| Self-Refine | `SQLITE_SELF_REFINE_TEMPLATE` | ⚠️ Needs data | `self_refine` |

### Current Default Training

The default config (`datasets_nl2sql_standard.json`) trains **Task 1 only** (Text-to-SQL).

This is sufficient for a solid baseline model. To get closer to the paper's performance, you would need to:

1. **Add Self-Refine data** (requires generating wrong predictions first)
2. **Implement Question/Evidence Inference** (requires new templates and data generation)

### How to Add Self-Refine Task (Advanced)

If you want to train on both Task 1 and Task 4:

1. Generate wrong predictions by running inference on training set
2. Create self-refine data with wrong SQL + errors
3. Update config to include both tasks:

```json
{
  "nl2sqlite_data": {
    "data_path": "data_warehouse/bird_train/processed_data/train_nl2sqlite.json",
    "task_name": "nl2sqlite",
    "sample_num": -1
  },
  "self_refine_data": {
    "data_path": "data_warehouse/bird_train/processed_data/train_self_refine.json",
    "task_name": "self_refine",
    "sample_num": -1
  }
}
```

---

## Architecture

### XiYan-SQL Full System Pipeline

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

**What This Repo Trains**: SQLG₁ (Task 1: Text-to-SQL, SQLite dialect)
**Not Implemented**: SQLG₂-₄ (formatting variations), Schema filtering at inference, Ensemble selection

### Repository Structure

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
│       └── datasets_nl2sql_standard.json
│
├── train/                   # Model training
│   ├── sft4xiyan.py         # Main training script
│   ├── xiyan_sft_3b.sh      # Training config (3B model)
│   ├── xiyan_sft.sh         # Training config (7B model)
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

### Key Components

**M-Schema** (`data/data_utils/m_schema.py`):
- Enhanced database schema representation
- Includes field types, constraints, comments, examples, and foreign keys
- Example:
  ```sql
  CREATE TABLE movies (
    movie_id INTEGER PRIMARY KEY,
    movie_title TEXT NOT NULL,
    rating_score REAL
  );
  -- Examples from movies:
  -- movie_id | movie_title | rating_score
  -- 1 | The Great Adventure | 8.5
  -- 2 | War Stories | 7.2
  ```

**Multi-Task Templates** (`data/data_utils/prompt_utils.py`):
- `NL2SQLITE_TEMPLATE` - Text-to-SQL for SQLite
- `SQLITE_SELF_REFINE_TEMPLATE` - SQL self-refinement
- `SQL2SELECT_TEMPLATE` - Candidate selection
- Templates for PostgreSQL, MySQL, Cypher, nGQL

**Training Script** (`train/xiyan_sft_3b.sh`):
- Pre-configured for Qwen2.5-Coder-3B
- Optimized hyperparameters (LR=2e-6, Batch=2)
- LoRA support (rank=512)
- DeepSpeed Zero2 integration

**Experiment Tracking** (SwanLab):
- Automatic logging of training metrics
- Real-time visualization dashboard
- Project: `SQLTrainer`, Experiment: `expr_id` parameter
- Local and cloud viewing options
- Commands:
  ```bash
  # Local dashboard
  swanlab watch train/output/dense/<expr_id>/swanlab

  # Cloud dashboard (after swanlab login)
  # Visit: https://swanlab.cn/@username/SQLTrainer
  ```

---

## Troubleshooting

### Out of Memory

**Solution 1**: Reduce batch size
```bash
# In xiyan_sft_3b.sh
BATCH_SIZE=1
ACC_STEP=4
```

**Solution 2**: Use fewer GPUs
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use only 4 GPUs
```

**Solution 3**: Reduce context length
```bash
MAX_LENGTH=8192  # Instead of 10240
```

### Training Too Slow

**Solution**: Increase batch size (if memory allows)
```bash
BATCH_SIZE=4
ACC_STEP=1
```

### Model Not Converging

**Solution 1**: Increase learning rate
```bash
LR=3e-6  # Instead of 2e-6
```

**Solution 2**: Train longer
```bash
EPOCH=7  # Instead of 5
```

---

## Contact Us

If you're interested in our research or products, please feel free to contact us.

#### Contact Information:
Yifu Liu, zhencang.lyf@alibaba-inc.com

#### Join Our DingTalk Group
<a href="https://github.com/alibaba/XiYan-SQL/XiYan-SQLTraining/blob/main/imgs/xiyansql_dingding.png">DingTalk Group</a>

---

## Applications

We welcome you to experience the intelligent query solutions developed based on XiYanSQL—**XiYan GBI**.

Log into Alibaba Cloud Bailian - Application Square - XiYan GBI. Any product experience and effect optimization suggestions are welcome for discussion.

- **Product introduction**: https://help.aliyun.com/zh/model-studio/user-guide/brief-introduction-of-gbi-products
- **Experience the product**: https://bailian.console.aliyun.com/xiyan
- **Product Ding Group**: 94725009401

---

## Citation

If you find our work helpful, we welcome you to cite us.

```bibtex
@article{XiYanSQL,
      title={XiYan-SQL: A Novel Multi-Generator Framework For Text-to-SQL},
      author={Yifu Liu and Yin Zhu and Yingqi Gao and Zhiling Luo and Xiaoxia Li and Xiaorong Shi and Yuntao Hong and Jinyang Gao and Yu Li and Bolin Ding and Jingren Zhou},
      year={2025},
      eprint={2507.04701},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.04701},
}
```

---

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
