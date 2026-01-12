#!/bin/bash

# Run baseline inference on pretrained model (no flash attention)

MODEL_PATH="../train/model/Qwen/Qwen2.5-Coder-3B-Instruct"
TEST_PATH="bird_evaluation/minidev_test.json"
EXPR_VERSION="baseline_pretrained"
BATCH_SIZE=4

echo "Running baseline inference on pretrained model..."
echo "Model: $MODEL_PATH"
echo "Test set: $TEST_PATH"
echo "Batch size: $BATCH_SIZE"
echo ""

# Modify the inference script temporarily to disable flash attention
python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')

# Monkey patch to disable flash attention check
import transformers
original_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained

def patched_from_pretrained(*args, **kwargs):
    # Remove flash attention params
    kwargs.pop('attn_implementation', None)
    return original_from_pretrained(*args, **kwargs)

transformers.AutoModelForCausalLM.from_pretrained = patched_from_pretrained

# Now run the evaluator
from sql_infer import Evaluator
import argparse

args = argparse.Namespace(
    model_name_or_path="../train/model/Qwen/Qwen2.5-Coder-3B-Instruct",
    lora_path="",
    expr_version="baseline_pretrained",
    test_set_path="bird_evaluation/minidev_test.json",
    batch_size=4,
    use_flash_attention=False,
    prompt_type="SQLite"
)

evaluator = Evaluator(args.model_name_or_path, args.lora_path, args.test_set_path, args.expr_version,
                      batch_size=args.batch_size, device='auto')
evaluator.model_init(use_flash_attention=False)
evaluator.inference_accelerator(temperature=0.01)

print("\n✅ Baseline inference completed!")
PYEOF
