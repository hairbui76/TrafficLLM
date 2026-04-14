# BitNet b1.58-2B-4T Integration

## Motivation

The original TrafficLLM relies on ChatGLM2-6B as its primary backbone. While effective, this creates several constraints:

- **Size:** 6B parameters require ~13GB VRAM for inference
- **Speed:** Standard FP16/BF16 inference is compute-bound
- **Deployment:** Difficult to deploy on edge devices or resource-constrained environments
- **Architecture lock-in:** P-Tuning v2 with `prefix_encoder` is ChatGLM-specific

Microsoft's BitNet b1.58-2B-4T offers a compelling alternative:

- **3x smaller** (2B vs 6B parameters)
- **Ternary weights** ({-1, 0, +1}) enable potential speedups via dedicated kernels
- **Llama-style architecture** compatible with standard HuggingFace PEFT
- **4T tokens** of pretraining data for strong general capabilities

## BitNet Technical Profile

| Specification | Value |
|---|---|
| Parameters | ~2 billion |
| Architecture | Llama-style decoder-only transformer |
| Layers | 30 |
| Hidden dimension | 2560 |
| Attention heads | 20 (5 KV heads via GQA) |
| Intermediate size | 6912 |
| Vocab size | 128,256 (LLaMA 3 tokenizer) |
| Max context | 4096 tokens |
| Weight quantization | Ternary {-1, 0, +1} (W1.58) via absmean quantization |
| Activation quantization | 8-bit (A8) |
| Position encoding | Rotary (RoPE) |
| Activation function | Squared ReLU (ReLU^2) |
| Normalization | Subln (no bias terms) |
| Training tokens | 4 trillion |
| HuggingFace ID | `microsoft/bitnet-b1.58-2B-4T` (inference) |
| BF16 variant | `microsoft/bitnet-b1.58-2B-4T-bf16` (training) |

## What Was Implemented

The `Adapt2BitNet/` module provides a complete training and inference pipeline for BitNet within TrafficLLM.

### Directory Structure

```
Adapt2BitNet/
├── FT/
│   ├── finetune.py              # LoRA fine-tuning script (560 lines)
│   ├── configs/
│   │   └── lora.yaml            # LoRA + training hyperparameters
│   └── scripts/
│       ├── trafficllm_stage1.sh # Stage 1: instruction tuning
│       └── trafficllm_stage2.sh # Stage 2: task-specific tuning
├── data/
│   └── convert_to_chat.py       # Data format converter (86 lines)
├── inference_bitnet.py          # Dual-stage inference (184 lines)
├── evaluation_bitnet.py         # Task evaluation (184 lines)
├── config.json                  # Model + adapter path config
├── requirements.txt             # Python dependencies
└── README.md                    # Module documentation
```

### File-by-File Breakdown

#### `FT/finetune.py`

Adapted from `Adapt2GLM4/FT/finetune.py`. Key changes:

- **Model loading:** Uses `AutoModelForCausalLM` with `torch_dtype=torch.bfloat16` -- no `empty_init` parameter (BitNet doesn't need it)
- **Tokenization:** Replaced GLM-4's hardcoded token IDs (`151331`, `151333`, `151337`) with `tokenizer.apply_chat_template()` for LLaMA 3 compatibility
- **Dual format support:** `process_batch()` handles both `messages` format (chat-style) and `instruction`/`output` format (original TrafficLLM), so data conversion is optional
- **Pad token:** Automatically sets `pad_token = eos_token` if not defined
- **Metrics:** Uses exact match and token-level F1 instead of ROUGE/BLEU (more appropriate for classification tasks)

Core function for building training inputs:

```python
def build_chat_input(tokenizer, user_content, assistant_content,
                     max_input_length, max_output_length):
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                              add_generation_prompt=False)
    user_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=True, add_generation_prompt=True)
    user_len = len(user_only)

    # Mask user tokens in labels
    labels = [-100] * min(user_len, len(full_ids))
    if len(full_ids) > user_len:
        labels += full_ids[user_len:]
    return full_ids, labels
```

#### `FT/configs/lora.yaml`

LoRA configuration targeting all 7 linear projection layers in BitNet's Llama-style architecture:

```yaml
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
```

Compared to GLM-4's single `query_key_value` target, BitNet's separate projections allow finer-grained LoRA control.

#### `inference_bitnet.py`

Implements dual-stage inference replacing ChatGLM2's `.chat()` pattern:

**ChatGLM2 pattern (before):**

```python
model.transformer.prefix_encoder.load_state_dict(prefix_weights)
response, history = model.chat(tokenizer, prompt, history=[])
```

**BitNet pattern (after):**

```python
model_with_adapter = PeftModel.from_pretrained(base_model, adapter_path)
messages = [{"role": "user", "content": prompt}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                           add_generation_prompt=True,
                                           return_tensors="pt")
outputs = model_with_adapter.generate(input_ids, max_new_tokens=128)
response = tokenizer.decode(outputs[0][input_ids.shape[1]:],
                            skip_special_tokens=True)
```

Key design: after Stage 1, the NLP adapter is unloaded via `model.unload()` before loading the task adapter, preventing memory accumulation.

#### `evaluation_bitnet.py`

Mirrors `evaluation.py` metrics (accuracy, precision, recall, F1, confusion matrix) but uses `model.generate()` instead of `.chat()`. Supports both detection and generation task evaluation.

#### `data/convert_to_chat.py`

Converts TrafficLLM's standard format to chat messages format:

```
Input:  {"instruction": "...", "output": "..."}
Output: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Handles both JSON array files and JSONL files. Preserves conversation history if present in the input.

## Architectural Comparison

```
ChatGLM2 Path                          BitNet Path
─────────────────                      ──────────────
AutoModel (Seq2Seq)                    AutoModelForCausalLM (CausalLM)
     │                                       │
config.pre_seq_len = 128               standard HF config
     │                                       │
model.half()                           model in bfloat16
prefix_encoder.float()                      │
     │                                       │
P-Tuning v2                           PEFT LoRA
(prefix_encoder weights)              (7 target modules)
     │                                       │
Manual state_dict swap                 PeftModel.from_pretrained()
per task                               + .unload() per task
     │                                       │
model.chat(tokenizer, prompt)          model.generate(input_ids)
     │                                 + apply_chat_template()
     ▼                                       ▼
response text                          response text
```

## LoRA Configuration Details

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `r` | 8 | Good balance of capacity vs. parameter count for classification |
| `lora_alpha` | 32 | 4x scaling factor for stable training |
| `lora_dropout` | 0.1 | Regularization for small datasets |
| `target_modules` | 7 modules | All linear projections in each transformer layer |
| `task_type` | `CAUSAL_LM` | Decoder-only causal language modeling |

With `r=8` across 7 modules in 30 layers, the trainable parameters are approximately:

```
7 modules x 30 layers x 2 x r x hidden = 7 x 30 x 2 x 8 x 2560 = ~8.6M parameters
```

This is ~0.4% of the total 2B parameters.

## Limitations and Caveats

### 1. No Ternary Speedup via Transformers

The efficiency benefits of BitNet's ternary weights (speed, energy, latency) require the dedicated C++ runtime [`bitnet.cpp`](https://github.com/microsoft/BitNet). When using the standard `transformers` library, BitNet operates on BF16 master weights with no special acceleration. The `transformers` path is suitable for:

- Training and fine-tuning
- Functional validation
- Accuracy benchmarking

For production inference with ternary speedups, export and use `bitnet.cpp`.

### 2. Smaller Model

At 2B parameters (vs ChatGLM2's 6B), BitNet has less capacity for complex traffic patterns. Empirical evaluation on each task is recommended before deployment.

### 3. Unofficial LoRA Support

Microsoft has not officially documented LoRA/PEFT fine-tuning for BitNet. The BF16 variant uses standard `nn.Linear` layers, so PEFT LoRA attaches normally, but edge cases may exist. Monitor training loss for stability.

### 4. Context Length

BitNet's 4096-token context length is sufficient for most traffic analysis tasks, but very long flow-level features may need truncation.

## Future Work

- **bitnet.cpp integration:** Export trained LoRA-merged weights to GGUF format for optimized ternary inference
- **Accuracy benchmarking:** Systematic comparison of BitNet vs ChatGLM2 across all 10 detection tasks
- **Streamlit integration:** Add BitNet as a selectable backend in `trafficllm_server.py`
- **Agent support:** Extend `agent/trafficllm_flask.py` to support BitNet backend
- **Quantization-aware training:** Explore training with ternary weight constraints instead of BF16 LoRA
