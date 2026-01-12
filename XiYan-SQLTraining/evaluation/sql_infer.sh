
MODEL_NAME_OR_PATH=${1}
EXPR_VERSION=${2}
TEST_SET_PATH=${3}
BATCH_SIZE=${4}
USE_FLASH_ATTENTION=${5}
MAX_SAMPLES=${6}

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"../train/model/Qwen/Qwen2.5-Coder-0.5B-Instruct"}
EXPR_VERSION=${EXPR_VERSION:-"test_0606"}
TEST_SET_PATH=${TEST_SET_PATH:-"bird_evaluation/eval_set/bird_dev_mschema_0926_short.json"}
BATCH_SIZE=${BATCH_SIZE:-"1"}
MAX_SAMPLES=${MAX_SAMPLES:-"None"}

# Build command with optional flash attention parameter
CMD="uv run sql_infer.py --model_name_or_path ${MODEL_NAME_OR_PATH} --expr_version ${EXPR_VERSION} --test_set_path ${TEST_SET_PATH} --batch_size ${BATCH_SIZE}"

# Add flash attention flag if requested (pass "true" or "1" to enable)
if [ "${USE_FLASH_ATTENTION}" = "true" ] || [ "${USE_FLASH_ATTENTION}" = "True" ] || [ "${USE_FLASH_ATTENTION}" = "1" ]; then
  CMD="${CMD} --use_flash_attention"
fi

# Add max samples parameter if provided
if [ "${MAX_SAMPLES}" != "None" ]; then
  CMD="${CMD} --max_samples ${MAX_SAMPLES}"
fi

CUDA_VISIBLE_DEVICES=0 ${CMD}

# Using accelerate launch for multi-GPU inference
#CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch --num_processes 2 --config_file config/zero2.yaml sql_infer.py --model_name_or_path ${MODEL_NAME_OR_PATH} --expr_version ${EXPR_VERSION} --test_set_path ${TEST_SET_PATH} --batch_size ${BATCH_SIZE}