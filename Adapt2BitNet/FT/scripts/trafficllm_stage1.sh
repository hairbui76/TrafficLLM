#!/bin/bash
# Stage 1: Instruction tuning for BitNet b1.58-2B-4T
# Trains LoRA adapters on instruction data for task understanding

NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=0

MODEL_DIR="microsoft/bitnet-b1.58-2B-4T-bf16"
DATA_DIR="../../datasets/instructions"
CONFIG_FILE="../configs/lora.yaml"
OUTPUT_DIR="../../models/bitnet/adapters/instruction"

# Convert instruction data to chat format if not already done
if [ ! -f "${DATA_DIR}/train.jsonl" ]; then
    echo "Converting instruction data to chat format..."
    python ../../data/convert_to_chat.py \
        --input_file "${DATA_DIR}/instructions.json" \
        --output_file "${DATA_DIR}/train.jsonl"
    cp "${DATA_DIR}/train.jsonl" "${DATA_DIR}/test.jsonl"
fi

# Run fine-tuning
torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS \
    ../finetune.py \
    "${DATA_DIR}" \
    "${MODEL_DIR}" \
    "${CONFIG_FILE}" \
    ""
