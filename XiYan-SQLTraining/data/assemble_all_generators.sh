#!/bin/bash

# This script assembles training datasets for all 4 XiYan-SQL generators (SQLG₁-₄)
# Each generator uses different augmentation strategies as per the paper

set -e  # Exit on error

echo "=========================================="
echo "Assembling datasets for 4 XiYan-SQL generators"
echo "=========================================="

# Create output directory if it doesn't exist
mkdir -p ../train/datasets

echo ""
echo "[1/4] Assembling SQLG₁ (Heavy Augmentation, aug_ratio=4.0)..."
bash data_assembler.sh \
  configs/datasets_sqlg1_heavy_aug.json \
  ../train/datasets/sqlg1_heavy_aug_train.json

echo ""
echo "[2/4] Assembling SQLG₂ (Moderate Augmentation, aug_ratio=2.0)..."
bash data_assembler.sh \
  configs/datasets_sqlg2_moderate_aug.json \
  ../train/datasets/sqlg2_moderate_aug_train.json

echo ""
echo "[3/4] Assembling SQLG₃ (Light Augmentation, aug_ratio=1.0)..."
bash data_assembler.sh \
  configs/datasets_sqlg3_light_aug.json \
  ../train/datasets/sqlg3_light_aug_train.json

echo ""
echo "[4/4] Assembling SQLG₄ (No Augmentation)..."
bash data_assembler.sh \
  configs/datasets_sqlg4_no_aug.json \
  ../train/datasets/sqlg4_no_aug_train.json

echo ""
echo "=========================================="
echo "✓ All 4 generator datasets assembled successfully!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - ../train/datasets/sqlg1_heavy_aug_train.json"
echo "  - ../train/datasets/sqlg2_moderate_aug_train.json"
echo "  - ../train/datasets/sqlg3_light_aug_train.json"
echo "  - ../train/datasets/sqlg4_no_aug_train.json"
echo ""
echo "Next steps:"
echo "1. Download base model: cd ../train/utils && uv run model_download.py"
echo "2. Train SQLG₁: Update xiyan_sft.sh with DATA=\"datasets/sqlg1_heavy_aug_train.json\""
echo "3. Repeat for SQLG₂, SQLG₃, SQLG₄"
echo ""
