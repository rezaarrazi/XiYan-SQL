#!/bin/bash
# XiYan-SQL Training Script for Qwen2.5-Coder-3B
# OPTIMIZED FOR COLAB 15GB GPU (T4)

# Set environment
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Basic distributed training configuration
GPUS_PER_NODE=$(uv run python -c "import torch; print(torch.cuda.device_count());")
MASTER_ADDR=${MASTER_ADDR:-localhost}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
MASTER_PORT=${MASTER_PORT:-12547}

# Use Zero2 for 15GB GPU
DS_CONFIG="config/zero2.yaml"

run_training() {
    local DATA=$1
    local OUTPUT=$2
    uv run accelerate launch --config_file $DS_CONFIG --num_machines $NNODES --num_processes $WORLD_SIZE --machine_rank $NODE_RANK --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
    sft4xiyan.py \
        --save_only_model True \
        --resume False \
        --model_name_or_path $MODEL \
        --data_path $DATA \
        --output_dir $OUTPUT \
        --num_train_epochs $EPOCH \
        --per_device_train_batch_size $BATCH_SIZE \
        --load_best_model_at_end False\
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps $ACC_STEP \
        --save_strategy "steps" \
        --eval_strategy "no" \
        --eval_steps $SAVE_STEP \
        --save_steps $SAVE_STEP \
        --save_total_limit 3 \
        --learning_rate $LR \
        --weight_decay $WEIGHT_DECAY \
        --adam_beta2 0.95 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type "cosine" \
        --log_level "info" \
        --logging_steps 10 \
        --report_to "none" \
        --model_max_length $MAX_LENGTH \
        --lazy_preprocess False \
        --gradient_checkpointing True \
        --predict_with_generate True \
        --include_inputs_for_metrics True \
        --use_lora $USE_LORA \
        --lora_r $LORA_R \
        --lora_alpha $((LORA_R * LORA_SCALE)) \
        --do_shuffle $SHUFFLE \
        --torch_compile False \
        --group_by_length $GROUP_BY_LENGTH \
        --model_type "auto" \
        --use_flash_attention True \
        --bf16 \
        --expr_id $EXPR_ID
}


# ============================================================================
# CONFIGURATION SECTION - OPTIMIZED FOR 15GB VRAM (Colab T4)
# ============================================================================

# Experiment ID
EXPR_ID="nl2sql_3b_colab_en"

# Model path - Qwen2.5-Coder-3B-Instruct
MODEL="model/Qwen/Qwen2.5-Coder-3B-Instruct"

# Training hyperparameters
EPOCH=3                  # Reduced for faster training in Colab
LR=2e-5                  # Learning rate
WEIGHT_DECAY=0.1         # Weight decay
MAX_LENGTH=8192          # Optimized for 15GB: 8192 tokens

# LoRA Configuration - Balanced for 15GB
USE_LORA=True            # Use LoRA for efficiency
LORA_R=128               # Higher rank for better quality on 15GB
LORA_SCALE=2             # Alpha multiplier

# Batch size and gradient accumulation - Optimized for 15GB
BATCH_SIZE=2             # Can use 2 with 15GB
ACC_STEP=16              # Effective batch size = 2 × 16 × 1 GPU = 32

SAVE_STEP=200            # Save checkpoints every 200 steps
GROUP_BY_LENGTH=True     # Group samples by length for efficiency
SHUFFLE=True             # Shuffle training data

# Dataset and output paths - Using English dataset
DATA="datasets/nl2sql_standard_train_en.json"  # English prompts
OUTPUT="output/dense/${EXPR_ID}/"

# ============================================================================
# RUN TRAINING
# ============================================================================

echo "=========================================="
echo "XiYan-SQL Training - COLAB 15GB MODE"
echo "=========================================="
echo "Experiment ID: $EXPR_ID"
echo "Model: $MODEL"
echo "Dataset: $DATA (English)"
echo "Output: $OUTPUT"
echo ""
echo "CONFIGURATION:"
echo "  - LoRA enabled: $USE_LORA (rank: $LORA_R)"
echo "  - Max length: $MAX_LENGTH tokens"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Gradient accumulation: $ACC_STEP"
echo "  - Effective batch size: $((BATCH_SIZE * ACC_STEP * WORLD_SIZE))"
echo "  - DeepSpeed: Zero2"
echo "  - Flash Attention: Enabled"
echo "  - Epochs: $EPOCH"
echo "  - Learning rate: $LR"
echo ""
echo "Expected VRAM usage: ~12-14GB"
echo "Training optimized for Colab T4 (15GB)!"
echo "=========================================="
echo ""

run_training $DATA $OUTPUT

echo ""
echo "=========================================="
echo "Training completed!"
echo "Model saved to: $OUTPUT"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Merge LoRA adapter:"
echo "   cd utils"
echo "   uv run adapter_merge.py \\"
echo "     --base_model ../model/Qwen/Qwen2.5-Coder-3B-Instruct \\"
echo "     --adapter ../output/dense/${EXPR_ID}/checkpoint-final \\"
echo "     --output ../merged_models/${EXPR_ID}"
echo ""
