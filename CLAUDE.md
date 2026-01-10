# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

XiYan-SQL is a natural language to SQL conversion framework that employs a multi-generator ensemble strategy. The repository includes the **XiYan-SQLTraining** framework for training SQL LLMs with capabilities for data processing, model training (including MOE multi-dialect models), and evaluation.

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

### Environment Setup
```bash
uv venv --python 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r XiYan-SQLTraining/requirements.txt
```

Requirements: PyTorch 2.3.1, transformers 4.42.3, DeepSpeed 0.12.0, PEFT 0.11.1, flash-attn 2.5.9

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
python model_download.py
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

**Merge LoRA adapter:**
```bash
python utils/adapter_merge.py
```

### Model Evaluation

**Run inference:**
```bash
cd XiYan-SQLTraining/evaluation
bash sql_infer.sh <model_path> <expr_version> <test_set_path> <batch_size>
```

**Evaluate results:**
```bash
bash sql_eval.sh <pred_sql_path> <test_sql_path> <db_conn_config> <save_eval_path>
```

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
- Raw data must include: `db_name`/`db_id`, `question`, and either `db_schema` or database connection info
- Processed data includes M-Schema with examples and metadata

## Architecture Notes

- The framework uses **conversation format** exclusively for training data
- **M-Schema is central** to XiYan's approach - it enriches standard schemas with examples and semantic information
- Training script automatically masks prompts, computing loss only on SQL generation
- MOE mode routes by `sql_type` field using first token position for dialect embedding
- DeepSpeed Zero2 is recommended for most use cases; Zero3 for very large models
- `group_by_length=True` recommended for efficiency with variable-length SQL queries
