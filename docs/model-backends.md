# Model Backends

TrafficLLM supports four LLM backends. Each has its own training pipeline, inference API, and adapter mechanism. This document compares them side by side.

## Comparison Table

| Aspect | ChatGLM2 | Llama | GLM-4 | BitNet b1.58 |
|--------|----------|-------|-------|--------------|
| **Parameters** | 6B | 7B+ | 9B | 2B |
| **Architecture** | Encoder-decoder-like (Seq2Seq) | Decoder-only (CausalLM) | Decoder-only (CausalLM) | Decoder-only (CausalLM, Llama-style) |
| **Model class** | `AutoModel` | `LlamaForCausalLM` | `AutoModelForCausalLM` | `AutoModelForCausalLM` |
| **PEFT method** | P-Tuning v2 (`prefix_encoder`) | LoRA / LlamaAdapter / Prefix | LoRA (via HF `peft`) | LoRA (via HF `peft`) |
| **Inference API** | `.chat()` | `model.generate()` | `model.generate()` | `model.generate()` + chat template |
| **Adapter loading** | Manual `prefix_state_dict` swap | `get_peft_model()` | `get_peft_model()` | `PeftModel.from_pretrained()` |
| **`transformers` version** | 4.30.2 | 4.30+ | 4.44.0 | >= 4.57 |
| **Data format** | JSONL (instruction/output) | JSONL (instruction/output) | JSONL (messages) | JSONL (instruction/output or messages) |
| **Training code** | `dual-stage-tuning/main.py` | `llm/llama-recipes/llama_finetuning.py` | `Adapt2GLM4/FT/finetune.py` | `Adapt2BitNet/FT/finetune.py` |
| **Status** | Primary (default) | Secondary (research) | Tertiary (faster GLM) | Experimental (ternary) |

## ChatGLM2 (Primary)

**Overview:** The default backbone for TrafficLLM, based on THUDM's ChatGLM2-6B. Uses a custom P-Tuning v2 mechanism with `prefix_encoder` for parameter-efficient fine-tuning.

**Key characteristics:**
- Uses `trust_remote_code=True` to load custom `modeling_chatglm.py`
- P-Tuning v2 adds learnable prefix tokens via `config.pre_seq_len = 128`
- The `prefix_encoder` stays in float32 while the rest of the model is in float16
- Inference uses the `.chat()` method with `history=[]`
- Checkpoints save only the `transformer.prefix_encoder.*` weights

**Files:**

| File | Purpose |
|------|---------|
| `inference.py` | Dual-stage CLI inference |
| `evaluation.py` | Task evaluation |
| `trafficllm_server.py` | Streamlit web demo |
| `dual-stage-tuning/main.py` | Seq2Seq training with prefix encoder |
| `dual-stage-tuning/arguments.py` | `ModelArguments` + `DataTrainingArguments` |
| `dual-stage-tuning/trainer_seq2seq.py` | Custom trainer with `save_changed` |
| `config.json` | Runtime configuration |

**Hardware:** ~13GB VRAM for inference (6B model in fp16 + prefix encoder in fp32).

**Dependencies:** `transformers==4.30.2`, `cpm_kernels`, `torch>=2.0`

## Llama (Secondary)

**Overview:** A research path using Meta's Llama 2 (or later) with LoRA fine-tuning. Located under `llm/llama-recipes/`, forked from Meta's official recipes with traffic dataset integration.

**Key characteristics:**
- Supports three PEFT methods: LoRA, LlamaAdapter, Prefix Tuning
- `llama_traffic.py` adds a classification head (sequence classification) instead of autoregressive generation
- Uses FSDP for multi-GPU training with Llama-specific layer wrapping
- LoRA config defaults: `r=8`, `alpha=32`, `dropout=0.05`

**Files:**

| File | Purpose |
|------|---------|
| `llm/llama-recipes/llama_finetuning.py` | Causal LM + PEFT + FSDP training |
| `llm/llama-recipes/llama_traffic.py` | Sequence classification + LoRA |
| `llm/llama-recipes/training_script.py` | Alternative training script |
| `llm/llama-recipes/configs/peft.py` | LoRA/Adapter/Prefix defaults |
| `llm/llama-recipes/configs/training.py` | Training hyperparameters |
| `llm/llama-recipes/ft_datasets/traffic_dataset.py` | Traffic dataset loader |

**Hardware:** ~16GB VRAM for 7B model with LoRA; multi-GPU via FSDP for larger models.

**Dependencies:** `transformers>=4.30`, `peft` (from git), `loralib`, `torch>=2.0`

## GLM-4 (Tertiary)

**Overview:** An adaptation path using THUDM's GLM-4-9B-Chat, which is faster than ChatGLM2 for both tuning and inference. Uses standard HuggingFace PEFT LoRA instead of custom P-Tuning.

**Key characteristics:**
- Uses `AutoModelForCausalLM` (decoder-only) instead of ChatGLM2's `AutoModel` (Seq2Seq)
- Standard HuggingFace `peft` library with LoRA via YAML config
- GLM-4 specific token IDs: `151331`, `151333`, `151337` for chat delimiters
- Supports `apply_chat_template()` for input formatting
- LoRA target: `query_key_value` (GLM-4's fused QKV projection)

**Files:**

| File | Purpose |
|------|---------|
| `Adapt2GLM4/FT/finetune.py` | CausalLM + PEFT LoRA training (Typer CLI) |
| `Adapt2GLM4/FT/inference.py` | GLM-4 inference |
| `Adapt2GLM4/FT/configs/lora.yaml` | LoRA + training hyperparameters |
| `Adapt2GLM4/FT/train.sh` | Training shell script |
| `Adapt2GLM4/FT/infer.sh` | Inference shell script |
| `Adapt2GLM4/Preprocess.py` | Data format converter |

**Hardware:** ~20GB VRAM for 9B model with LoRA in bf16.

**Dependencies:** `transformers==4.44.0`, `peft`, `torch>=2.0`

## BitNet b1.58-2B-4T (Experimental)

**Overview:** Microsoft's ternary LLM with 2B parameters, using weights quantized to {-1, 0, +1}. Integrated into TrafficLLM as a lightweight alternative to the 6B+ models. Uses the BF16 master weight variant for training.

**Key characteristics:**
- Llama-style architecture: 30 layers, 2560 hidden, 20 attention heads (5 KV heads via GQA)
- Ternary quantization (W1.58A8) -- but training uses BF16 master weights
- Standard `AutoModelForCausalLM` loading, no `trust_remote_code` needed (transformers >= 4.57)
- LLaMA 3 tokenizer with 128K vocabulary
- LoRA targets all 7 linear projection layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Inference via `model.generate()` with `tokenizer.apply_chat_template()`
- LoRA adapter hot-swap using `PeftModel.from_pretrained()` / `.unload()`

**Files:**

| File | Purpose |
|------|---------|
| `Adapt2BitNet/FT/finetune.py` | LoRA fine-tuning (adapted from GLM-4 path) |
| `Adapt2BitNet/FT/configs/lora.yaml` | LoRA + training hyperparameters |
| `Adapt2BitNet/FT/scripts/trafficllm_stage1.sh` | Stage 1 instruction tuning |
| `Adapt2BitNet/FT/scripts/trafficllm_stage2.sh` | Stage 2 task-specific tuning |
| `Adapt2BitNet/inference_bitnet.py` | Dual-stage inference with adapter swap |
| `Adapt2BitNet/evaluation_bitnet.py` | Task evaluation |
| `Adapt2BitNet/data/convert_to_chat.py` | Data format converter |
| `Adapt2BitNet/config.json` | Model + adapter path configuration |

**Hardware:** ~8GB VRAM for inference (2B model in bf16), ~16GB for LoRA training.

**Dependencies:** `transformers>=4.57`, `peft>=0.7.0`, `accelerate`, `torch>=2.0`

**Limitations:**
1. Standard `transformers` does NOT provide ternary inference speedups -- use `bitnet.cpp` for that
2. Smaller model (2B vs 6B) may trade accuracy for efficiency
3. LoRA on BitNet BF16 weights is not officially documented by Microsoft

## Choosing a Backend

| Use Case | Recommended Backend |
|----------|-------------------|
| Paper reproduction, default deployment | ChatGLM2 |
| Classification tasks with labeled data | Llama (via `llama_traffic.py`) |
| Faster training/inference than ChatGLM2 | GLM-4 |
| Resource-constrained environments | BitNet |
| Edge deployment research | BitNet (with `bitnet.cpp`) |
| Multi-GPU distributed training | Llama (via FSDP) |
