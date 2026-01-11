#!/bin/bash
# XiYan-SQL Training Script for Qwen2.5-Coder-3B
# OPTIMIZED FOR LOW MEMORY (8GB VRAM - RTX 3080 Ti)

# Set NCCL_IB as needed
# export NCCL_P2P_DISABLE=1  # Uncomment if you have P2P issues

# wandb has been deprecated, now use swanlab
#export WANDB_PROJECT='xiyan-sql-3b'
export CUDA_VISIBLE_DEVICES=0  # Single GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Reduce fragmentation

# Basic distributed training configuration
GPUS_PER_NODE=$(uv run python -c "import torch; print(torch.cuda.device_count());")
MASTER_ADDR=${MASTER_ADDR:-localhost}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
MASTER_PORT=${MASTER_PORT:-12547}

# Use Zero3 with CPU offload for maximum memory efficiency
DS_CONFIG="config/zero3_offload.yaml"

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
        --save_total_limit 100 \
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
        --use_flash_attention False \
        --bf16 \
        --expr_id $EXPR_ID
        # --eval_data_path $EVAL_DATA
}


# ============================================================================
# CONFIGURATION SECTION - OPTIMIZED FOR 8GB VRAM
# ============================================================================

# Experiment ID
EXPR_ID="nl2sql_3b_lowmem"

# Model path - Qwen2.5-Coder-3B-Instruct
MODEL="model/Qwen/Qwen2.5-Coder-3B-Instruct"

# Training hyperparameters - Adjusted for low memory
EPOCH=5                  # Number of epochs
LR=2e-6                  # Learning rate
WEIGHT_DECAY=0.1         # Weight decay
MAX_LENGTH=2048          # FURTHER REDUCED: 2048 for 8GB (was 4096)

# LoRA Configuration - REQUIRED for 8GB VRAM
USE_LORA=True            # MUST be True for 8GB VRAM
LORA_R=32                # FURTHER REDUCED: 32 for 8GB (was 64)
LORA_SCALE=2             # Increased alpha to compensate

# Batch size and gradient accumulation - CRITICAL for 8GB VRAM
BATCH_SIZE=1             # MINIMUM: Must be 1 for 8GB
ACC_STEP=64              # FURTHER INCREASED: Compensate with higher accumulation
                         # Effective batch size = 1 × 64 × 1 GPU = 64

SAVE_STEP=100            # Save more frequently (fewer steps per epoch)
GROUP_BY_LENGTH=True     # Group samples by length for efficiency
SHUFFLE=True             # Shuffle training data

# Dataset and output paths
DATA="datasets/nl2sql_standard_train_en.json"  # English prompts
OUTPUT="output/dense/${EXPR_ID}/"

# Optional: Enable evaluation during training
#EVAL_DATA="datasets/eval_set.json"

# ============================================================================
# RUN TRAINING
# ============================================================================

echo "=========================================="
echo "XiYan-SQL Training - LOW MEMORY MODE"
echo "=========================================="
echo "Experiment ID: $EXPR_ID"
echo "Model: $MODEL"
echo "Dataset: $DATA"
echo "Output: $OUTPUT"
echo ""
echo "MEMORY OPTIMIZATIONS:"
echo "  - LoRA enabled: $USE_LORA (rank: $LORA_R)"
echo "  - Max length: $MAX_LENGTH (VERY LOW for 8GB)"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Gradient accumulation: $ACC_STEP"
echo "  - Effective batch size: $((BATCH_SIZE * ACC_STEP * WORLD_SIZE))"
echo "  - DeepSpeed: Zero3 with CPU offload"
echo "  - Flash Attention: Disabled"
echo "  - CUDA Memory: Expandable segments enabled"
echo ""
echo "Expected VRAM usage: ~6-7GB on RTX 3080 Ti"
echo "Training will be SLOW but should fit in memory!"
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
