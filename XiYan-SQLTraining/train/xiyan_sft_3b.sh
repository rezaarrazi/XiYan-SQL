#!/bin/bash
# XiYan-SQL Training Script for Qwen2.5-Coder-3B
# Optimized configuration for 3B model

# Set NCCL_IB as needed
# export NCCL_P2P_DISABLE=1  # Uncomment if you have P2P issues

# wandb has been deprecated, now use swanlab
#export WANDB_PROJECT='xiyan-sql-3b'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Basic distributed training configuration
GPUS_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count());")
MASTER_ADDR=${MASTER_ADDR:-localhost}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
MASTER_PORT=${MASTER_PORT:-12547}

# Use Zero2 for 3B model (more efficient than Zero3 for smaller models)
DS_CONFIG="config/zero2.yaml"

run_training() {
    local DATA=$1
    local OUTPUT=$2
    accelerate launch --config_file $DS_CONFIG --num_machines $NNODES --num_processes $WORLD_SIZE --machine_rank $NODE_RANK --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
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
        --use_flash_attention True \
        --bf16 \
        --expr_id $EXPR_ID
        # --eval_data_path $EVAL_DATA
}


# ============================================================================
# CONFIGURATION SECTION - Edit these parameters for your training
# ============================================================================

# Experiment ID - Change this for each training run
EXPR_ID="nl2sql_3b_standard"

# Model path - Qwen2.5-Coder-3B-Instruct
MODEL="model/Qwen/Qwen2.5-Coder-3B-Instruct"

# Training hyperparameters - Optimized for 3B model
EPOCH=5                  # Number of epochs
LR=2e-6                  # Learning rate (slightly higher for smaller model)
WEIGHT_DECAY=0.1         # Weight decay for regularization
MAX_LENGTH=10240         # Maximum sequence length

# LoRA Configuration
# Set USE_LORA=True for memory efficiency (recommended for multi-GPU training)
# Set USE_LORA=False for full fine-tuning (better results but needs more memory)
USE_LORA=True            # Enable LoRA
LORA_R=512               # LoRA rank (512 for high capacity)
LORA_SCALE=1             # LoRA alpha scaling factor

# Batch size and gradient accumulation
# For 3B model on A100 40GB: can use larger batch size than 7B
BATCH_SIZE=2             # Per-device batch size (can increase for 3B)
ACC_STEP=2               # Gradient accumulation steps
                         # Effective batch size = BATCH_SIZE × ACC_STEP × NUM_GPUS
                         # Example: 2 × 2 × 8 = 32

SAVE_STEP=500            # Save checkpoint every N steps
GROUP_BY_LENGTH=True     # Group samples by length for efficiency
SHUFFLE=True             # Shuffle training data

# Dataset and output paths
DATA="datasets/nl2sql_standard_train.json"
OUTPUT="output/dense/${EXPR_ID}/"

# Optional: Enable evaluation during training
#EVAL_DATA="datasets/eval_set.json"

# ============================================================================
# RUN TRAINING
# ============================================================================

echo "=========================================="
echo "XiYan-SQL Training - Qwen2.5-Coder-3B"
echo "=========================================="
echo "Experiment ID: $EXPR_ID"
echo "Model: $MODEL"
echo "Dataset: $DATA"
echo "Output: $OUTPUT"
echo "Learning Rate: $LR"
echo "Epochs: $EPOCH"
echo "Batch Size: $BATCH_SIZE"
echo "Accumulation Steps: $ACC_STEP"
echo "Effective Batch Size: $((BATCH_SIZE * ACC_STEP * WORLD_SIZE))"
echo "LoRA Enabled: $USE_LORA"
if [ "$USE_LORA" = "True" ]; then
    echo "LoRA Rank: $LORA_R"
fi
echo "GPUs: $WORLD_SIZE"
echo "=========================================="
echo ""

run_training $DATA $OUTPUT

echo ""
echo "=========================================="
echo "Training completed!"
echo "Model saved to: $OUTPUT"
echo "=========================================="
