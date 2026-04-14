# Training Guide

## Prerequisites

### Environment Setup

```bash
conda create -n trafficllm python=3.9
conda activate trafficllm
cd TrafficLLM
pip install -r requirements.txt
pip install rouge_chinese nltk jieba datasets
```

### Data Preparation

1. Download datasets from [Google Drive](https://drive.google.com/drive/folders/1RZAOPcNKq73-quA8KG_lkAo_EqlwhlQb) or preprocess your own:

```bash
cd preprocess
python preprocess_dataset.py \
    --input /path/to/raw/pcaps \
    --dataset_name ustc-tfc-2016 \
    --traffic_task detection \
    --granularity packet \
    --output_path ../datasets/ustc-tfc-2016 \
    --output_name ustc-tfc-2016_detection_packet
```

2. For BitNet/GLM-4, convert to chat format:

```bash
python Adapt2BitNet/data/convert_to_chat.py \
    --input_file datasets/ustc-tfc-2016/ustc-tfc-2016_detection_packet_train.json \
    --output_file datasets/ustc-tfc-2016_chat/train.jsonl
```

### Model Weights

- **ChatGLM2:** Download from [HuggingFace](https://huggingface.co/THUDM/chatglm2-6b) to `models/chatglm2/chatglm2-6b/`
- **GLM-4:** Download from [HuggingFace](https://huggingface.co/THUDM/glm-4-9b-chat)
- **Llama:** Download from [Meta](https://huggingface.co/meta-llama)
- **BitNet:** Auto-downloaded from HuggingFace as `microsoft/bitnet-b1.58-2B-4T-bf16`

---

## ChatGLM2 Training

Uses P-Tuning v2 with `pre_seq_len=128` prefix tokens. Only the `prefix_encoder` weights are trained.

### Stage 1: Instruction Tuning

```bash
cd dual-stage-tuning
bash trafficllm_stage1.sh
```

Or manually:

```bash
PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=0

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file ../datasets/instructions/instructions.json \
    --validation_file ../datasets/instructions/instructions.json \
    --preprocessing_num_workers 10 \
    --prompt_column instruction \
    --response_column output \
    --overwrite_cache \
    --cache_dir ../cache \
    --model_name_or_path ../models/chatglm2/chatglm2-6b \
    --output_dir ../models/chatglm2/peft/instruction \
    --overwrite_output_dir \
    --max_source_length 1024 \
    --max_target_length 32 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 20000 \
    --logging_steps 10 \
    --save_steps 4000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN
```

### Stage 2: Task-Specific Tuning

```bash
bash trafficllm_stage2.sh
```

Change `--train_file` and `--output_dir` per task:

| Task | Train File | Output Dir |
|------|-----------|------------|
| MTD | `ustc-tfc-2016_detection_packet_train.json` | `peft/ustc-tfc-2016-detection-packet` |
| BND | `iscx-botnet-2014_detection_packet_train.json` | `peft/iscx-botnet-2014-detection-packet` |
| EVD | `iscx-vpn-2016_detection_packet_train.json` | `peft/iscx-vpn-2016-detection-packet` |
| TBD | `iscx-tor-2016_detection_packet_train.json` | `peft/iscx-tor-2016-detection-packet` |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| `pre_seq_len` | 128 |
| `learning_rate` | 2e-2 |
| `max_steps` | 20000 |
| `batch_size` | 1 |
| `gradient_accumulation_steps` | 16 |
| `max_source_length` | 1024 |
| `max_target_length` | 32 |

---

## Llama Training

Uses LoRA with optional FSDP for multi-GPU training.

### Causal LM Fine-tuning

```bash
cd llm/llama-recipes
python llama_finetuning.py \
    --model_name /path/to/llama-2-7b \
    --peft_method lora \
    --dataset traffic_dataset \
    --output_dir output/
```

### Sequence Classification (LoRA)

```bash
python training_script.py \
    --data_path preprocess/build_datasets/detection/ustc-tfc-2016 \
    --model_name /path/to/llama-2-7b \
    --num_epochs 20 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --lr 0.02 \
    --max_length 512 \
    --lora_rank 16 \
    --lora_alpha 64 \
    --lora_dropout 0.074 \
    --weight_decay 0.006 \
    --set_pad_id
```

### LoRA Defaults

| Parameter | Value |
|-----------|-------|
| `r` | 8 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.05 |
| `target_modules` | model-dependent |

---

## GLM-4 Training

Uses PEFT LoRA via YAML config file. Requires `transformers==4.44.0`.

### Setup

```bash
cd Adapt2GLM4/FT
pip install -r requirements.txt
```

### Training

```bash
bash train.sh
```

Or manually:

```bash
python finetune.py \
    /path/to/data_dir \
    /path/to/glm-4-9b-chat \
    configs/lora.yaml
```

### LoRA Config (`Adapt2GLM4/FT/configs/lora.yaml`)

```yaml
peft_config:
  peft_type: LORA
  task_type: CAUSAL_LM
  r: 8
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules: ["query_key_value"]

training_args:
  max_steps: 3000
  bf16: true
  learning_rate: 5e-5
  per_device_train_batch_size: 1
  save_steps: 500
```

---

## BitNet Training

Uses PEFT LoRA on the BF16 master weight variant of BitNet b1.58-2B-4T. Requires `transformers>=4.57`.

### Setup

```bash
cd Adapt2BitNet
pip install -r requirements.txt
```

### Stage 1: Instruction Tuning

```bash
cd FT/scripts
bash trafficllm_stage1.sh
```

This script:
1. Converts instruction data to chat format (if not already done)
2. Runs `torchrun` with `finetune.py` using the LoRA config

### Stage 2: Task-Specific Tuning

```bash
bash trafficllm_stage2.sh <dataset_name> <train_file>
```

Examples:

```bash
# Malware Traffic Detection
bash trafficllm_stage2.sh ustc-tfc-2016-detection-packet \
    ../../datasets/ustc-tfc-2016/ustc-tfc-2016_detection_packet_train.json

# Botnet Detection
bash trafficllm_stage2.sh iscx-botnet-2014-detection-packet \
    ../../datasets/iscx-botnet-2014/iscx-botnet-2014_detection_packet_train.json

# VPN Detection
bash trafficllm_stage2.sh iscx-vpn-2016-detection-packet \
    ../../datasets/iscx-vpn-2016/iscx-vpn-2016_detection_packet_train.json
```

### LoRA Config (`Adapt2BitNet/FT/configs/lora.yaml`)

```yaml
peft_config:
  peft_type: LORA
  task_type: CAUSAL_LM
  r: 8
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

training_args:
  max_steps: 10000
  bf16: true
  learning_rate: 5e-5
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  warmup_steps: 200
  weight_decay: 0.01
  save_steps: 2000
  max_input_length: 1024
  max_output_length: 128
```

### Tuning Tips

- **Detection tasks:** `max_output_length: 32` is sufficient since outputs are short labels
- **Generation tasks:** increase `max_output_length` to 512 for hex payload generation
- **More capacity:** increase `r` from 8 to 16 or 32 for complex multi-class tasks
- **Memory constraints:** reduce `per_device_train_batch_size` to 1 and increase `gradient_accumulation_steps`
- **Resume training:** pass `yes` as the 4th argument to `finetune.py` to auto-resume from latest checkpoint

---

## Training Output

All backends produce checkpoints in subdirectories:

- **ChatGLM2:** `models/chatglm2/peft/<task>/checkpoint-<step>/pytorch_model.bin` (prefix weights only)
- **BitNet:** `models/bitnet/adapters/<task>/` (full LoRA adapter with `adapter_model.safetensors`)
- **GLM-4:** `output/checkpoint-<step>/` (LoRA adapter)
- **Llama:** `output/` (LoRA adapter or full model)

Register checkpoints in the appropriate `config.json` to use them for inference.
