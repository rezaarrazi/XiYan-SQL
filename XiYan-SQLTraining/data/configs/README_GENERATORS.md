# XiYan-SQL Generator Configurations

This directory contains dataset configurations for training the 4 XiYan-SQL generators (SQLG₁-₄) with different augmentation strategies.

## Configuration Files

| File | Generator | Augmentation | Description |
|------|-----------|--------------|-------------|
| `datasets_sqlg1_heavy_aug.json` | SQLG₁ | Heavy (4.0x) | Maximum diversity, 4 augmented samples per original |
| `datasets_sqlg2_moderate_aug.json` | SQLG₂ | Moderate (2.0x) | Balanced augmentation, 2 samples per original |
| `datasets_sqlg3_light_aug.json` | SQLG₃ | Light (1.0x) | Minimal augmentation, 1 sample per original |
| `datasets_sqlg4_no_aug.json` | SQLG₄ | None | No augmentation, original samples only |

## Augmentation Techniques Applied

When `data_aug: true`, the following augmentation operations are applied:

1. **SchemaShuffle**: Randomize table/column order
2. **SchemaFilter**: Simulate schema filtering by removing irrelevant parts
3. **SchemaPermute**: Reorder schema elements differently
4. **SQLTranslate**: Translate between SQL dialects (if applicable)

The `aug_ratio` parameter controls how many augmented samples are generated per original sample.

## Quick Start

### Option 1: Assemble All Generators at Once

```bash
cd XiYan-SQLTraining/data
bash assemble_all_generators.sh
```

This will create 4 training datasets in `../train/datasets/`:
- `sqlg1_heavy_aug_train.json`
- `sqlg2_moderate_aug_train.json`
- `sqlg3_light_aug_train.json`
- `sqlg4_no_aug_train.json`

### Option 2: Assemble Individually

```bash
cd XiYan-SQLTraining/data

# SQLG₁ - Heavy augmentation
bash data_assembler.sh \
  configs/datasets_sqlg1_heavy_aug.json \
  ../train/datasets/sqlg1_heavy_aug_train.json

# SQLG₂ - Moderate augmentation
bash data_assembler.sh \
  configs/datasets_sqlg2_moderate_aug.json \
  ../train/datasets/sqlg2_moderate_aug_train.json

# SQLG₃ - Light augmentation
bash data_assembler.sh \
  configs/datasets_sqlg3_light_aug.json \
  ../train/datasets/sqlg3_light_aug_train.json

# SQLG₄ - No augmentation
bash data_assembler.sh \
  configs/datasets_sqlg4_no_aug.json \
  ../train/datasets/sqlg4_no_aug_train.json
```

## Training Each Generator

After assembling the datasets, train each generator:

```bash
cd ../train

# Download base model (if not already done)
cd utils
uv run model_download.py
# Select: Qwen2.5-Coder-7B-Instruct
cd ..

# Train SQLG₁
# Edit xiyan_sft.sh:
#   EXPR_ID="sqlg1_heavy_aug"
#   DATA="datasets/sqlg1_heavy_aug_train.json"
#   OUTPUT="output/dense/sqlg1_heavy_aug/"
bash xiyan_sft.sh

# Train SQLG₂
# Edit xiyan_sft.sh:
#   EXPR_ID="sqlg2_moderate_aug"
#   DATA="datasets/sqlg2_moderate_aug_train.json"
#   OUTPUT="output/dense/sqlg2_moderate_aug/"
bash xiyan_sft.sh

# Train SQLG₃
# Edit xiyan_sft.sh:
#   EXPR_ID="sqlg3_light_aug"
#   DATA="datasets/sqlg3_light_aug_train.json"
#   OUTPUT="output/dense/sqlg3_light_aug/"
bash xiyan_sft.sh

# Train SQLG₄
# Edit xiyan_sft.sh:
#   EXPR_ID="sqlg4_no_aug"
#   DATA="datasets/sqlg4_no_aug_train.json"
#   OUTPUT="output/dense/sqlg4_no_aug/"
bash xiyan_sft.sh
```

## Configuration Parameters

### Required Fields

- **`data_path`**: Path to processed data file (relative to `data/` directory)
- **`sample_num`**: Number of samples to use (-1 = use all)
- **`sum_num`**: Total number of samples in the dataset (informational)
- **`task_name`**: Task type (e.g., `nl2sqlite`, `nl2pgsql`, `nl2mysql`)
- **`data_aug`**: Enable/disable augmentation (`true` or `false`)

### Optional Fields

- **`aug_ratio`**: Augmentation multiplier (default: 1.0)
  - `1.0` = 1 augmented sample per original
  - `2.0` = 2 augmented samples per original
  - `4.0` = 4 augmented samples per original
  - Only applies when `data_aug: true`

## Expected Dataset Sizes

Based on BIRD train data (221 samples) + test data (3 samples):

| Generator | Base Samples | After Augmentation | Total Size |
|-----------|--------------|-------------------|------------|
| SQLG₁ (4.0x) | 224 | 224 × 4 = 896 | ~896 samples |
| SQLG₂ (2.0x) | 224 | 224 × 2 = 448 | ~448 samples |
| SQLG₃ (1.0x) | 224 | 224 × 1 = 224 | ~224 samples |
| SQLG₄ (0x) | 224 | No augmentation | 224 samples |

**Note**: Actual sizes may vary based on augmentation randomness and filtering.

## Adding More Data Sources

To include additional datasets (e.g., Spider, BIRD dev, custom data):

1. Process the raw data first:
   ```bash
   bash data_processing.sh \
     data_warehouse/spider_train/raw_data/train.json \
     data_warehouse/spider_train/db_conn.json \
     data_warehouse/spider_train/processed_data/ \
     data_warehouse/spider_train/mschema/ \
     configs/datasets_all.json
   ```

2. Add entry to each generator config:
   ```json
   {
     "bird_train_processed_data_train_nl2sqlite": { ... },
     "spider_train_processed_data_train_nl2sqlite": {
       "data_path": "data_warehouse/spider_train/processed_data/train_nl2sqlite.json",
       "sample_num": -1,
       "sum_num": 7000,
       "task_name": "nl2sqlite",
       "data_aug": true,
       "aug_ratio": 4.0
     }
   }
   ```

## Multi-Dialect Training

To train generators for multiple SQL dialects, create separate configs:

```json
{
  "bird_train_nl2sqlite": {
    "data_path": "data_warehouse/bird_train/processed_data/train_nl2sqlite.json",
    "task_name": "nl2sqlite",
    "data_aug": true,
    "aug_ratio": 2.0
  },
  "bird_train_nl2pgsql": {
    "data_path": "data_warehouse/bird_train/processed_data/train_nl2pgsql.json",
    "task_name": "nl2pgsql",
    "data_aug": true,
    "aug_ratio": 2.0
  },
  "bird_train_nl2mysql": {
    "data_path": "data_warehouse/bird_train/processed_data/train_nl2mysql.json",
    "task_name": "nl2mysql",
    "data_aug": true,
    "aug_ratio": 2.0
  }
}
```

For multi-dialect training, consider using the MOMQ MOE training script (`xiyan_momq_sft.sh`) instead.

## Troubleshooting

**Issue**: "FileNotFoundError: data_warehouse/bird_train/processed_data/train_nl2sqlite.json"

**Solution**: Run data processing first:
```bash
bash data_processing.sh \
  data_warehouse/bird_train/raw_data/train.json \
  data_warehouse/bird_train/db_conn.json \
  data_warehouse/bird_train/processed_data/ \
  data_warehouse/bird_train/mschema/ \
  configs/datasets_all.json
```

**Issue**: "Out of memory during augmentation"

**Solution**: Reduce `aug_ratio` or process datasets in smaller batches by setting `sample_num` to a lower value.

## References

- Main Documentation: [`../TRAINING_STRATEGIES.md`](../../TRAINING_STRATEGIES.md)
- Quick Start Guide: [`../QUICK_START.md`](../../QUICK_START.md)
- XiYan-SQL Paper: [arXiv:2507.04701](https://arxiv.org/abs/2507.04701)
