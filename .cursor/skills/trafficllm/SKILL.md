# TrafficLLM Project Skill

## Overview

TrafficLLM is a framework that adapts large language models for network traffic analysis tasks including malware detection, botnet detection, web attack detection, APT attack detection, encrypted VPN detection, and Tor behavior detection. It uses a **dual-stage tuning** approach: Stage 1 identifies the user's intended task from natural language instructions, and Stage 2 applies task-specific adapters to classify traffic data.

Paper: arXiv:2504.04222

## Architecture

```text
TrafficLLM/
├── preprocess/                    # PCAP → JSONL data pipeline (model-agnostic)
├── dual-stage-tuning/             # ChatGLM2 P-Tuning v2 training
│   ├── main.py                    # Seq2Seq trainer with prefix_encoder
│   ├── arguments.py               # ModelArguments + DataTrainingArguments
│   ├── trainer_seq2seq.py         # Custom Seq2SeqTrainer (save_changed for prefix)
│   ├── trafficllm_stage1.sh      # Instruction tuning script
│   └── trafficllm_stage2.sh      # Task-specific tuning script
├── llm/llama-recipes/             # Llama LoRA/PEFT fine-tuning (secondary path)
│   ├── llama_finetuning.py        # LlamaForCausalLM + PEFT (LoRA/Adapter/Prefix)
│   ├── llama_traffic.py           # Llama sequence classification + LoRA
│   ├── configs/                   # training, peft, fsdp, datasets configs
│   └── ft_datasets/               # traffic_dataset.py
├── Adapt2GLM4/                    # GLM-4 LoRA fine-tuning path
│   └── FT/
│       ├── finetune.py            # AutoModelForCausalLM + PEFT LoRA
│       └── configs/lora.yaml      # LoRA hyperparameters
├── Adapt2BitNet/                  # BitNet b1.58-2B-4T ternary LLM path
│   ├── FT/
│   │   ├── finetune.py            # BitNet bf16 + PEFT LoRA training
│   │   ├── configs/lora.yaml      # LoRA config for BitNet layers
│   │   └── scripts/               # Stage 1 & 2 training shell scripts
│   ├── inference_bitnet.py        # Dual-stage inference with LoRA hot-swap
│   ├── evaluation_bitnet.py       # Task evaluation metrics
│   ├── data/convert_to_chat.py    # instruction/output → chat messages converter
│   └── config.json                # Model and adapter paths
├── models/
│   └── chatglm2/peft/            # Released P-Tuning v2 checkpoints
├── inference.py                   # ChatGLM2 dual-stage CLI inference
├── evaluation.py                  # ChatGLM2 task evaluation
├── trafficllm_server.py           # Streamlit demo (ChatGLM2)
├── config.json                    # Runtime config (model_path, peft_set, tasks)
└── requirements.txt               # Root dependencies
```

## Model Paths

### ChatGLM2 (Primary / Default)

- **Type**: Encoder-decoder-like (Seq2Seq), 6B params
- **PEFT**: P-Tuning v2 via `prefix_encoder` (not HuggingFace `peft` library)
- **Inference**: `AutoModel` + `.chat()` API
- **Config**: `config.pre_seq_len = 128`, `transformer.prefix_encoder`
- **Files**: `inference.py`, `evaluation.py`, `trafficllm_server.py`, `dual-stage-tuning/main.py`

### Llama (Secondary)

- **Type**: Decoder-only (CausalLM)
- **PEFT**: LoRA / LlamaAdapter / Prefix via HuggingFace `peft`
- **Files**: `llm/llama-recipes/llama_finetuning.py`, `llama_traffic.py`

### GLM-4 (Tertiary)

- **Type**: Decoder-only (CausalLM)
- **PEFT**: LoRA via HuggingFace `peft` + YAML config
- **Files**: `Adapt2GLM4/FT/finetune.py`, `Adapt2GLM4/FT/configs/lora.yaml`

### BitNet b1.58-2B-4T (Ternary LLM)

- **Type**: Decoder-only (CausalLM), Llama-style, 2B params
- **Quantization**: Ternary weights {-1, 0, +1}, 8-bit activations
- **PEFT**: LoRA via HuggingFace `peft` on bf16 variant
- **Inference**: `AutoModelForCausalLM` + `model.generate()` + chat template
- **Files**: `Adapt2BitNet/FT/finetune.py`, `Adapt2BitNet/inference_bitnet.py`
- **Requires**: `transformers >= 4.57`, `peft >= 0.7.0`

## Dual-Stage Pipeline

```text
User Instruction + Traffic Data
        │
        ▼
   ┌─────────────┐
   │   Stage 1    │  NLP adapter identifies the task
   │  (Instruction│  e.g., "Malware Traffic Detection" → "MTD"
   │   Adapter)   │
   └──────┬──────┘
          │ task code
          ▼
   ┌─────────────┐
   │   Stage 2    │  Task-specific adapter classifies traffic
   │(Task Adapter)│  e.g., MTD adapter → "Cridex" / "Zeus" / ...
   └──────┬──────┘
          │
          ▼
     Final Result
```

## Data Pipeline

1. **Raw PCAPs** → `preprocess/preprocess_dataset.py` → JSONL with `instruction` + `output` columns
2. For ChatGLM2: JSONL used directly with `prompt_column=instruction`, `response_column=output`
3. For BitNet/GLM-4: Convert to chat messages format via `Adapt2BitNet/data/convert_to_chat.py`

## Task Registry

Defined in `config.json` → `tasks` map:

| Human Name | Code | Dataset |
| --- | --- | --- |
| Malware Traffic Detection | MTD | USTC-TFC-2016 |
| Botnet Detection | BND | ISCX-Botnet-2014 |
| Web Attack Detection | WAD | CSIC-2010 |
| APT Attack Detection | AAD | DAPT-2020 |
| Encrypted VPN Detection | EVD | ISCX-VPN-2016 |
| Tor Behavior Detection | TBD | ISCX-Tor-2016 |

## Key Patterns

### Adding a New Model Backbone

1. Create `Adapt2<ModelName>/` directory following `Adapt2GLM4/` or `Adapt2BitNet/` structure
2. Implement `finetune.py` using `AutoModelForCausalLM` + `get_peft_model()` + LoRA
3. Create inference script replacing `.chat()` with `model.generate()` + chat template
4. Create data converter if tokenizer expects a different input format
5. Update `requirements.txt` for any new transformers version requirements

### Adding a New Traffic Task

1. Preprocess new dataset via `preprocess/preprocess_dataset.py`
2. Add preprompt template in inference scripts
3. Add task code to `config.json` under `tasks` and `peft_set`/`adapter_set`
4. Run Stage 2 fine-tuning for the new task

## Dependencies

- **ChatGLM2 path**: `transformers==4.30.2`, `cpm_kernels`
- **BitNet path**: `transformers>=4.57`, `peft>=0.7.0`, `accelerate`
- **Common**: `torch>=2.0`, `scikit-learn`, `scapy`, `streamlit`
