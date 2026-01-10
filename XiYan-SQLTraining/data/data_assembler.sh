DATASET_CONFIG_PATH=${1}
SAVE_PATH=${2}
DATASET_CONFIG_PATH=${DATASET_CONFIG_PATH:-"configs/datasets_example.json"}
SAVE_PATH=${SAVE_PATH:-"output/train_examples.json"}
uv run data_assembler.py --dataset_config_path ${DATASET_CONFIG_PATH} --save_path ${SAVE_PATH}