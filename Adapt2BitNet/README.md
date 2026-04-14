# Adapt2BitNet: BitNet b1.58-2B-4T for TrafficLLM

This module integrates Microsoft's [BitNet b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) ternary LLM into the TrafficLLM framework, replacing ChatGLM2/Llama as the backbone model.

## Architecture

BitNet b1.58-2B-4T is a 2-billion parameter LLM with ternary weights ({-1, 0, +1}) and 8-bit activations. It uses a Llama-style decoder-only transformer architecture:

- **Parameters**: 2B (vs ChatGLM2's 6B)
- **Layers**: 30, Hidden: 2560, Attention Heads: 20 (5 KV heads, GQA)
- **Tokenizer**: LLaMA 3 (128K vocab)
- **Context**: 4096 tokens
- **Training variant**: `microsoft/bitnet-b1.58-2B-4T-bf16` (BF16 master weights)

### Key Changes from ChatGLM2 Path

| Aspect | ChatGLM2 | BitNet |
| --- | --- | --- |
| Model class | `AutoModel` (Seq2Seq) | `AutoModelForCausalLM` (CausalLM) |
| PEFT method | P-Tuning v2 (`prefix_encoder`) | LoRA (on all linear layers) |
| Inference API | `.chat()` | `model.generate()` + chat template |
| Adapter loading | Manual `prefix_state_dict` swap | `PeftModel.from_pretrained()` |

## Setup

```bash
pip install -r requirements.txt
```

**Requirements**: Python 3.10+, CUDA 11.8+, ~8GB VRAM for inference, ~16GB for LoRA training.

> **Note**: Standard `transformers` execution does NOT provide the ternary inference speed benefits described in the BitNet paper. For optimized inference, use [bitnet.cpp](https://github.com/microsoft/BitNet). The `transformers` path is used for training and functional validation.

## Quick Start

### 1. Data Preparation

Convert existing TrafficLLM datasets from instruction/output format to chat messages format:

```bash
python data/convert_to_chat.py \
    --input_file ../datasets/instructions/instructions.json \
    --output_file ../datasets/instructions/train.jsonl
```

### 2. Stage 1: Instruction Tuning

Train the NLP adapter for task understanding:

```bash
cd FT/scripts
bash trafficllm_stage1.sh
```

Or manually:

```bash
cd FT
torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    finetune.py \
    ../../datasets/instructions_chat \
    microsoft/bitnet-b1.58-2B-4T-bf16 \
    configs/lora.yaml \
    ""
```

### 3. Stage 2: Task-Specific Tuning

Train task adapters for each traffic analysis task:

```bash
cd FT/scripts

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

### 4. Inference

Run dual-stage inference:

```bash
python inference_bitnet.py \
    --config config.json \
    --prompt "Please help me detect malware traffic.<packet>frame.encap_type: 1, frame.time: Mar 15 2024, ..."
```

### 5. Evaluation

Evaluate on test datasets:

```bash
python evaluation_bitnet.py \
    --model_name microsoft/bitnet-b1.58-2B-4T-bf16 \
    --test_file ../datasets/ustc-tfc-2016/ustc-tfc-2016_detection_packet_test.json \
    --label_file ../datasets/ustc-tfc-2016/ustc-tfc-2016_detection_packet_label.json \
    --adapter_path models/bitnet/adapters/ustc-tfc-2016-detection-packet \
    --traffic_task detection
```

## File Structure

```
Adapt2BitNet/
├── FT/
│   ├── finetune.py              # LoRA fine-tuning script
│   ├── configs/
│   │   └── lora.yaml            # LoRA + training hyperparameters
│   └── scripts/
│       ├── trafficllm_stage1.sh # Stage 1 instruction tuning
│       └── trafficllm_stage2.sh # Stage 2 task-specific tuning
├── data/
│   └── convert_to_chat.py       # Data format converter
├── inference_bitnet.py          # Dual-stage inference
├── evaluation_bitnet.py         # Task evaluation metrics
├── config.json                  # Model + adapter path config
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## LoRA Configuration

The default LoRA config targets all linear projection layers in the BitNet architecture:

```yaml
peft_type: LORA
task_type: CAUSAL_LM
r: 8
lora_alpha: 32
lora_dropout: 0.1
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

**Tuning tips**:
- Increase `r` to 16 or 32 for more capacity on complex tasks (at the cost of more parameters)
- For detection tasks, `max_output_length: 32` is sufficient
- For generation tasks, increase `max_output_length` to 512

## Limitations

1. **No ternary speedup via transformers**: The efficiency gains from 1.58-bit weights require the dedicated `bitnet.cpp` runtime. Training and evaluation via HuggingFace transformers operates on the BF16 master weights.
2. **Smaller model**: 2B vs 6B parameters may affect accuracy on complex traffic patterns. Empirical evaluation is recommended.
3. **PEFT/LoRA on BitNet**: While LoRA works on the BF16 variant's standard linear layers, this is not officially documented by Microsoft. Monitor training loss for stability.
