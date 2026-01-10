# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

XiYan-SQL is a natural language to SQL conversion framework that employs a multi-generator ensemble strategy. This repository (**XiYan-SQLTraining**) implements the **training component** that creates the SQL generators used in the full system.

### Full XiYan-SQL Pipeline

The complete XiYan-SQL system operates in 4 stages:
1. **Inputs**: Question, Evidence (hints), Full Database Schema
2. **Schema Filter**: Intelligently filter schema to relevant tables/columns
3. **Multiple SQL Generation**: 4 generators (SQLG₁-₄) produce SQL candidates
4. **SQL Selection**: Ensemble voting/selection to choose best SQL

**This repository trains the generators for Stage 3.** Schema filtering at inference time and ensemble selection are separate components not included here.

**📖 For detailed training strategies and implementation guide, see [`XiYan-SQLTraining/TRAINING_STRATEGIES.md`](XiYan-SQLTraining/TRAINING_STRATEGIES.md)**

## Key Architecture Components

### 1. Data Pipeline (`XiYan-SQLTraining/data/`)

The data processing pipeline operates in two main stages:

**Stage 1: Raw Data Processing** (`data_processing.py`)
- Converts raw SQL datasets into processable format
- Generates **M-Schema** (metadata-enhanced schema) from database connections
- M-Schema includes table/column descriptions, examples, data types, and foreign key relationships
- Uses `SchemaEngine` to extract schema information directly from databases
- Outputs processed data to warehouse folders and optionally saves to config registry

**Stage 2: Data Assembly** (`data_assembler.py`)
- Packages multiple processed datasets into final training format
- Supports data augmentation (schema augmentation, prompt variations)
- Produces conversation format: `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "SELECT..."}]`
- Each sample includes `sql_type` field for multi-dialect support (sqlite, postgresql, mysql, etc.)

### 2. Training Framework (`XiYan-SQLTraining/train/`)

**Two Training Modes:**

**Standard SFT** (`xiyan_sft.sh` + `sft4xiyan.py`)
- Fine-tunes base models (primarily Qwen2.5-Coder series) for Text-to-SQL
- Supports full fine-tuning or LoRA
- Uses DeepSpeed (Zero1/2/3) with accelerate for distributed training
- Chat template masking: only assistant responses are used for loss computation

**MOMQ MOE Training** (`xiyan_momq_sft.sh`)
- XiYan's Multi-Dialect MOE architecture
- Uses LoRA experts routed by token or dialect
- Includes dialect router with auxiliary loss for multi-dialect specialization
- Supports PostgreSQL, MySQL, SQLite, Cypher, NGQL dialects
- Label smoothing and load balancing options available

**Training Data Format:**
```json
{
  "id": 0,
  "conversations": [
    {"role": "user", "content": "You are an SQLite expert..."},
    {"role": "assistant", "content": "SELECT ..."}
  ],
  "sql_type": "sqlite"
}
```

### 3. Evaluation Framework (`XiYan-SQLTraining/evaluation/`)

**Two-Step Process:**

1. **Inference** (`sql_infer.py`): Generates SQL predictions from model
2. **Evaluation** (`sql_eval.py`): Compares predictions against ground truth using database execution

Evaluation connects to actual databases to verify SQL correctness (execution accuracy).

### 4. Key Technical Components

**M-Schema** (`data/data_utils/m_schema.py`)
- Enhanced database schema representation
- Includes field types, constraints, comments, examples, and categories
- Foreign key relationships preserved
- Supports schema filtering and augmentation

**MOE/MOMQ Architecture** (`train/trainer/moe_momq.py`)
- LoRA-based Mixture of Experts
- Token-level or dialect-level routing
- Dialect router for multi-SQL-dialect specialization
- Custom model configurations for Qwen2 and Mistral

**Custom Models** (`train/model/`)
- Modified Qwen2 and Mistral architectures
- Support for MOE-LoRA integration
- Flash attention support

## Common Development Commands

### System Requirements

**Platform:**
- **Linux required** - DeepSpeed requires Linux-specific libraries (libaio) and has symlink permission issues on Windows
- **Windows users must use WSL2** (Windows Subsystem for Linux 2)

**CUDA:**
- **Minimum: CUDA 12.6** (PyTorch 2.9.0 is built for CUDA 12.6)
- CUDA versions 11.8-12.4 may work with older PyTorch versions but are not officially supported in current setup
- Verify your CUDA version: `nvcc --version` or `nvidia-smi`

**Python:**
- Python 3.10 or higher (specified in pyproject.toml)

### Environment Setup

**Using uv (recommended):**
```bash
cd XiYan-SQLTraining
uv sync
```
All scripts use `uv run` to automatically use the correct virtual environment. No manual activation needed.

**Using conda (alternative):**
```bash
conda create -n xiyansql python=3.10
conda activate xiyansql
cd XiYan-SQLTraining
uv pip install -e .  # Install from pyproject.toml
```
With conda, you must activate the environment before running any scripts.

**Key Dependencies:**
- PyTorch 2.9.0+cu126 (CUDA 12.6)
- transformers 4.57.3+
- DeepSpeed 0.18.4+ (Linux only)
- PEFT 0.18.1+
- accelerate 1.12.0+
- SwanLab 0.7.6+
- flash-attn (built without isolation, configured in pyproject.toml)

### Data Processing

**Process raw data into warehouse:**
```bash
cd XiYan-SQLTraining/data
bash data_processing.sh <raw_data_path> <db_conn_config> <processed_data_dir> <save_mschema_dir> <save_to_configs>
```

**Assemble training data:**
```bash
bash data_assembler.sh <dataset_config_path> <save_path>
```

### Model Training

**Download model:**
```bash
cd XiYan-SQLTraining/train/utils
uv run model_download.py
```

**Standard SFT training:**
```bash
cd XiYan-SQLTraining/train
bash xiyan_sft.sh
```

**MOMQ MOE training:**
```bash
bash xiyan_momq_sft.sh
```

**Multi-node distributed training:**
Set environment variables before running training scripts:
```bash
export MASTER_ADDR=<master_node_ip>
export MASTER_PORT=12547
export WORLD_SIZE=<total_num_nodes>
export RANK=<current_node_rank>  # 0 for master, 1,2,... for workers
```

**Merge LoRA adapter:**
```bash
cd XiYan-SQLTraining/train
uv run utils/adapter_merge.py
```

### Model Evaluation

**Run inference (single GPU):**
```bash
cd XiYan-SQLTraining/evaluation
bash sql_infer.sh <model_path> <expr_version> <test_set_path> <batch_size>
```

**Run inference (multi-GPU with accelerate):**
```bash
cd XiYan-SQLTraining/evaluation
CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch --num_processes 2 --config_file config/zero2.yaml \
  sql_infer.py --model_name_or_path <model_path> --expr_version <version> \
  --test_set_path <test_set> --batch_size <batch_size>
```

**Evaluate results:**
```bash
bash sql_eval.sh <pred_sql_path> <test_sql_path> <db_conn_config> <save_eval_path>
```
This connects to actual databases and executes both predicted and ground-truth SQL to verify correctness.

## Important Configuration Files

- `train/config/*.yaml`: DeepSpeed configurations (zero1/2/3, offload variants)
- `data/configs/datasets_*.json`: Dataset registry and configuration
- Database connection configs: `db_conn.json` files in warehouse folders

## Key Parameters

**Training:**
- `model_max_length`: Context length (default 8192-10240)
- `use_lora`: Enable LoRA (recommended for large models)
- `use_moe_lora`: Enable MOMQ MOE mode
- `enable_dialect_router`: Enable dialect-specific routing
- `num_experts`, `num_experts_per_tok`: MOE configuration

**Data:**
- Raw data format: JSON list with entries containing `db_name`/`db_id`, `question`, and either `db_schema` or database connection info
- Processed data includes M-Schema with examples and metadata
- Final training data: conversation format with `id`, `conversations` array (user/assistant roles), and `sql_type` field
- Example training data available at `train/datasets/train_examples.json`

## Architecture Notes

- The framework uses **conversation format** exclusively for training data
- **M-Schema is central** to XiYan's approach - it enriches standard schemas with examples and semantic information
- Training script automatically masks prompts, computing loss only on SQL generation
- MOE mode routes by `sql_type` field using first token position for dialect embedding
- DeepSpeed Zero2 is recommended for most use cases; Zero3 for very large models
- `group_by_length=True` recommended for efficiency with variable-length SQL queries
- SwanLab has replaced wandb for experiment tracking (configure via environment variables if needed)
- Standard SFT uses `torch_compile=False`, MOMQ uses `torch_compile=True` by default
- to memorize
- to memorize