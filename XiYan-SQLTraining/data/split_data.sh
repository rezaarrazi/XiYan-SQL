#!/bin/bash

# split_data.sh
# Split processed data into train and test splits
#
# Usage:
#   bash split_data.sh <input_file> <output_dir> [train_ratio] [seed] [train_config] [test_config] [--stratified] [--reindex] [--data_aug]
#
# Example:
#   bash split_data.sh \
#     data_warehouse/train/processed_data/train_nl2sqlite.json \
#     data_warehouse/train/processed_data/ \
#     0.8 \
#     42 \
#     configs/train_only.json \
#     configs/test_only.json \
#     --stratified \
#     --reindex

INPUT_FILE=${1}
OUTPUT_DIR=${2}
TRAIN_RATIO=${3:-0.8}
SEED=${4}
TRAIN_CONFIG=${5}
TEST_CONFIG=${6}

# Default values
INPUT_FILE=${INPUT_FILE:-"data_warehouse/bird_train/processed_data/train_nl2sqlite.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"data_warehouse/bird_train/processed_data/"}

# Build command
CMD="uv run split_data.py --input_file ${INPUT_FILE} --output_dir ${OUTPUT_DIR} --train_ratio ${TRAIN_RATIO}"

# Add seed if provided
if [ -n "${SEED}" ] && [[ ! "${SEED}" =~ ^-- ]]; then
    CMD="${CMD} --seed ${SEED}"
fi

# Add train config if provided
if [ -n "${TRAIN_CONFIG}" ] && [[ ! "${TRAIN_CONFIG}" =~ ^-- ]]; then
    CMD="${CMD} --train_config ${TRAIN_CONFIG}"
fi

# Add test config if provided
if [ -n "${TEST_CONFIG}" ] && [[ ! "${TEST_CONFIG}" =~ ^-- ]]; then
    CMD="${CMD} --test_config ${TEST_CONFIG}"
fi

# Parse flags from all arguments
for arg in "$@"; do
    case $arg in
        --stratified)
            CMD="${CMD} --stratified"
            ;;
        --reindex)
            CMD="${CMD} --reindex"
            ;;
        --data_aug)
            CMD="${CMD} --data_aug"
            ;;
    esac
done

# Execute
echo "Running: ${CMD}"
${CMD}
