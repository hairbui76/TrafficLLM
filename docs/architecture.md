# Architecture

## System Overview

TrafficLLM adapts large language models for network traffic analysis through three core techniques:

1. **Traffic-Domain Tokenization** -- extends LLM tokenizers with traffic-specific vocabulary trained on large-scale traffic corpora.
2. **Dual-Stage Tuning Pipeline** -- Stage 1 trains the LLM to understand task instructions; Stage 2 trains task-specific adapters on traffic data.
3. **EA-PEFT (Extensible Adaptation with Parameter-Efficient Fine-Tuning)** -- organizes PEFT adapters so new tasks can be registered or existing ones updated with minimal overhead.

```
                        ┌──────────────────────────────────────────────────────────────┐
                        │                     TrafficLLM Framework                     │
                        └──────────────────────────────────────────────────────────────┘

  ┌────────────┐     ┌──────────────────┐     ┌──────────────────────────────────────┐
  │  Raw PCAP  │────>│  Preprocessing   │────>│  JSONL (instruction + output)        │
  │  Files     │     │  preprocess/     │     │  or chat messages format             │
  └────────────┘     └──────────────────┘     └──────────┬───────────────────────────┘
                                                         │
                          ┌──────────────────────────────┼──────────────────────────┐
                          │                              │                          │
                          ▼                              ▼                          ▼
                ┌──────────────────┐          ┌────────────────────┐     ┌────────────────────┐
                │   ChatGLM2 (6B)  │          │   Llama (7B+)      │     │  BitNet (2B)       │
                │   P-Tuning v2    │          │   LoRA / Adapter   │     │  LoRA on bf16      │
                │   dual-stage-    │          │   llm/llama-       │     │  Adapt2BitNet/     │
                │   tuning/        │          │   recipes/         │     │                    │
                └────────┬─────────┘          └────────────────────┘     └──────────┬─────────┘
                         │                                                          │
                         ▼                                                          ▼
                ┌──────────────────┐                                     ┌────────────────────┐
                │  Prefix Encoder  │                                     │  LoRA Adapters      │
                │  Checkpoints     │                                     │  (per task)         │
                │  models/chatglm2 │                                     │  models/bitnet/     │
                │  /peft/          │                                     │  adapters/          │
                └────────┬─────────┘                                     └──────────┬─────────┘
                         │                                                          │
                         ▼                                                          ▼
                ┌──────────────────────────────────────────────────────────────────────┐
                │                     Dual-Stage Inference                             │
                │  Stage 1: Instruction Understanding → task code (e.g., "MTD")        │
                │  Stage 2: Task-Specific Analysis   → result (e.g., "Cridex")         │
                └──────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
                ┌──────────────────────────────────────────────────────────────────────┐
                │  Deployment: CLI / Streamlit / Flask API / Qwen-Agent                │
                └──────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
TrafficLLM/
├── config.json                     # Root runtime config: model paths, PEFT dirs, task registry
├── requirements.txt                # Root Python dependencies (ChatGLM2 stack)
├── inference.py                    # ChatGLM2 dual-stage CLI inference
├── evaluation.py                   # ChatGLM2 task evaluation (accuracy, F1, confusion matrix)
├── trafficllm_server.py            # Streamlit web demo (ChatGLM2 backend)
├── trafficllm_server_text.py       # Streamlit variant (text-oriented UI)
│
├── preprocess/                     # PCAP → JSONL data pipeline (model-agnostic)
│   ├── preprocess_dataset.py       # CLI entry point (--input, --dataset_name, --traffic_task, etc.)
│   ├── preprocess_utils.py         # Train/test split, JSONL writing, label file generation
│   ├── packet_data_preprocess.py   # Packet-level feature extraction via Scapy
│   ├── flow_data_preprocess.py     # Flow-level feature extraction
│   └── specfic_dataset_utils.py    # Dataset-specific logic (e.g., USTC-TFC-2016 folder layout)
│
├── dual-stage-tuning/              # ChatGLM2 P-Tuning v2 training
│   ├── main.py                     # HuggingFace Seq2Seq trainer with prefix_encoder
│   ├── arguments.py                # ModelArguments (pre_seq_len, prefix_projection) + DataTrainingArguments
│   ├── trainer_seq2seq.py          # Custom Seq2SeqTrainer with save_changed for prefix weights
│   ├── trafficllm_stage1.sh        # Stage 1: instruction tuning shell script
│   └── trafficllm_stage2.sh        # Stage 2: task-specific traffic tuning shell script
│
├── EA-PEFT/                        # Extensible PEFT orchestration
│   └── ea-peft.py                  # Automates update/register of PEFT adapters
│
├── tokenization/                   # Traffic-domain tokenizer
│   └── traffic_tokenizer.py        # SentencePiece BPE training + comparison vs ChatGLM2 tokenizer
│
├── llm/                            # Alternative LLM backends
│   ├── README.md                   # Llama2 + DeepSeek-R1 usage notes
│   └── llama-recipes/              # Fork of Meta llama-recipes with traffic extensions
│       ├── llama_finetuning.py     # LlamaForCausalLM + PEFT (LoRA/Adapter/Prefix) + FSDP
│       ├── llama_traffic.py        # Llama + LoRA for sequence classification
│       ├── training_script.py      # Alternative LoRA seq-cls script
│       ├── configs/                # training.py, peft.py, fsdp.py, datasets.py
│       ├── ft_datasets/            # traffic_dataset.py
│       ├── inference/              # Llama inference scripts
│       └── docs/                   # single_gpu.md, multi_gpu.md, inference.md, FAQ.md
│
├── Adapt2GLM4/                     # GLM-4 LoRA fine-tuning path
│   ├── README.md
│   ├── Preprocess.py               # Data format converter for GLM-4
│   └── FT/
│       ├── finetune.py             # AutoModelForCausalLM + PEFT LoRA via YAML
│       ├── inference.py            # GLM-4 inference
│       ├── configs/lora.yaml       # LoRA hyperparameters (target: query_key_value)
│       ├── train.sh                # Training shell script
│       ├── infer.sh                # Inference shell script
│       └── requirements.txt        # GLM-4 specific deps (transformers 4.44.0)
│
├── Adapt2BitNet/                   # BitNet b1.58-2B-4T ternary LLM path
│   ├── README.md
│   ├── config.json                 # Model path + adapter paths + task registry
│   ├── requirements.txt            # BitNet deps (transformers >= 4.57, peft >= 0.7.0)
│   ├── inference_bitnet.py         # Dual-stage inference with LoRA adapter hot-swap
│   ├── evaluation_bitnet.py        # Task evaluation (accuracy, F1, confusion matrix)
│   ├── data/
│   │   └── convert_to_chat.py      # instruction/output → chat messages format converter
│   └── FT/
│       ├── finetune.py             # BitNet bf16 + PEFT LoRA training
│       ├── configs/lora.yaml       # LoRA config (7 target modules)
│       └── scripts/
│           ├── trafficllm_stage1.sh
│           └── trafficllm_stage2.sh
│
├── agent/                          # Agent systems via Qwen-Agent
│   ├── README.md
│   ├── trafficllm_flask.py         # Flask API serving TrafficLLM as HTTP endpoint
│   ├── assistant_qwen3.py          # Single-agent demo with registered tools
│   └── group_chat.py               # Multi-agent demo (malware expert, VPN expert, etc.)
│
├── tutorials/                      # Traffic generation with Scapy
│   ├── README.md
│   ├── config.json                 # Tutorial config (includes generation task checkpoints)
│   └── generation.py               # Generate pcap files from LLM-produced hex data
│
├── models/                         # Model weights and checkpoints (downloaded separately)
│   └── chatglm2/
│       ├── chatglm2-6b/            # Base ChatGLM2 weights (from HuggingFace)
│       └── peft/                   # Per-task P-Tuning v2 checkpoints
│           ├── instruction/        # Stage 1 NLP adapter
│           ├── ustc-tfc-2016-.../  # Stage 2 per-dataset adapters
│           └── ...
│
├── images/                         # Figures for README (framework diagram, web demo GIF)
│
└── docs/                           # This documentation directory
```

## Dual-Stage Pipeline

The core innovation of TrafficLLM is its dual-stage inference pipeline that separates **task understanding** from **task execution**.

### Stage 1: Instruction Understanding

The user provides a natural language instruction describing what traffic analysis task to perform (e.g., "Please help me detect malware traffic"). The NLP adapter processes this instruction and maps it to one of the registered task codes.

**Input:** "Please help me detect malware traffic"
**Output:** "Malware Traffic Detection"
**Mapped to:** `MTD`

### Stage 2: Task-Specific Traffic Analysis

Using the task code from Stage 1, the system loads the corresponding task adapter and processes the actual traffic data with a task-specific preprompt template.

**Input:** Preprompt template + traffic data (protocol fields, features, payloads)
**Output:** Classification label (e.g., "Cridex", "Zeus", "normal")

### How Adapters Are Swapped

- **ChatGLM2**: Manually loads `prefix_encoder` state dict from `pytorch_model.bin` per task
- **BitNet**: Uses `PeftModel.from_pretrained()` and `.unload()` for clean LoRA adapter hot-swap

## Configuration Files

### Root `config.json` (ChatGLM2)

```json
{
    "model_path": "models/chatglm2/chatglm2-6b/",
    "peft_path": "models/chatglm2/peft/",
    "peft_set": {
        "NLP": "instruction/checkpoint-8000/",
        "MTD": "ustc-tfc-2016-detection-packet/checkpoint-10000/",
        "BND": "iscx-botnet-2014-detection-packet/checkpoint-5000/",
        ...
    },
    "tasks": {
        "Malware Traffic Detection": "MTD",
        "Botnet Detection": "BND",
        ...
    }
}
```

### `Adapt2BitNet/config.json`

```json
{
    "model_path": "microsoft/bitnet-b1.58-2B-4T-bf16",
    "adapter_path": "models/bitnet/adapters/",
    "adapter_set": {
        "NLP": "instruction/",
        "MTD": "ustc-tfc-2016-detection-packet/",
        ...
    },
    "tasks": { ... }
}
```

Key differences: BitNet uses `adapter_path`/`adapter_set` (LoRA adapters) instead of `peft_path`/`peft_set` (prefix checkpoints), and the model path points to a HuggingFace model ID rather than a local directory.

## Task Registry

All tasks from the paper, across detection and generation:

| Mainstream | Downstream Task | Code | Dataset | Samples |
|---|---|---|---|---|
| Detection | Malware Traffic Detection | MTD | USTC-TFC-2016 | 50.7K |
| Detection | Botnet Detection | BND | ISCX-Botnet-2014 | 25.0K |
| Detection | Malicious DoH Detection | MDD | DoHBrw-2020 | 47.8K |
| Detection | Web Attack Detection | WAD | CSIC-2010 | 34.5K |
| Detection | APT Attack Detection | AAD | DAPT-2020 | 10.0K |
| Detection | Encrypted VPN Detection | EVD | ISCX-VPN-2016 | 64.8K |
| Detection | Tor Behavior Detection | TBD | ISCX-Tor-2016 | 40.0K |
| Detection | Encrypted App Classification | EAC | CSTNET-2023 | 97.6K |
| Detection | Website Fingerprinting | WF | CW-100-2018 | 7.4K |
| Detection | Concept Drift | CD | APP-53-2023 | 109.8K |
| Generation | Malware Traffic Generation | MTG | USTC-TFC-2016 | -- |
| Generation | Botnet Traffic Generation | BTG | ISCX-Botnet-2014 | -- |
| Generation | Encrypted VPN Generation | EVG | ISCX-VPN-2016 | -- |
| Generation | Encrypted App Generation | EAG | CSTNET-2023 | -- |

The instruction dataset contains 9,209 task-specific instructions across all tasks, used for Stage 1 training.
