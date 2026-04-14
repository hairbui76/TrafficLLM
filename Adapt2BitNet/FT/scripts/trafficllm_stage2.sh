#!/bin/bash
# Stage 2: Task-specific traffic tuning for BitNet b1.58-2B-4T
# Trains LoRA adapters on specific traffic analysis datasets
#
# Usage: ./trafficllm_stage2.sh <dataset_name> <train_file>
# Example: ./trafficllm_stage2.sh ustc-tfc-2016-detection-packet ../datasets/ustc-tfc-2016/ustc-tfc-2016_detection_packet_train.json

NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=0

MODEL_DIR="microsoft/bitnet-b1.58-2B-4T-bf16"
CONFIG_FILE="../configs/lora.yaml"

DATASET_NAME=${1:-"ustc-tfc-2016-detection-packet"}
TRAIN_FILE=${2:-"../../datasets/ustc-tfc-2016/ustc-tfc-2016_detection_packet_train.json"}
OUTPUT_DIR="../../models/bitnet/adapters/${DATASET_NAME}"
DATA_DIR="../../datasets/${DATASET_NAME}_chat"

# Convert dataset to chat format
mkdir -p "${DATA_DIR}"
if [ ! -f "${DATA_DIR}/train.jsonl" ]; then
    echo "Converting ${DATASET_NAME} data to chat format..."
    python ../../data/convert_to_chat.py \
        --input_file "${TRAIN_FILE}" \
        --output_file "${DATA_DIR}/train.jsonl"
    cp "${DATA_DIR}/train.jsonl" "${DATA_DIR}/test.jsonl"
fi

# Create output-specific config with correct output_dir
TASK_CONFIG="/tmp/bitnet_stage2_${DATASET_NAME}.yaml"
sed "s|output_dir: ./output|output_dir: ${OUTPUT_DIR}|g" "${CONFIG_FILE}" > "${TASK_CONFIG}"

# Run fine-tuning
torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS \
    ../finetune.py \
    "${DATA_DIR}" \
    "${MODEL_DIR}" \
    "${TASK_CONFIG}" \
    ""
