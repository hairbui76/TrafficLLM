# Developer Guide

## Adding a New Model Backbone

Follow the established `Adapt2<Name>/` pattern used by GLM-4 and BitNet.

### Step-by-Step

#### 1. Create the module directory

```
Adapt2<Name>/
├── FT/
│   ├── finetune.py
│   ├── configs/
│   │   └── lora.yaml
│   └── scripts/
│       ├── trafficllm_stage1.sh
│       └── trafficllm_stage2.sh
├── data/
│   └── convert_to_chat.py      # if model needs chat format
├── inference_<name>.py
├── evaluation_<name>.py
├── config.json
├── requirements.txt
└── README.md
```

#### 2. Determine the model architecture

| Architecture | Model class | PEFT method | Inference API |
|---|---|---|---|
| Seq2Seq | `AutoModel` | P-Tuning v2 | `.chat()` (custom) |
| CausalLM | `AutoModelForCausalLM` | LoRA | `model.generate()` |
| Seq2SeqLM | `AutoModelForSeq2SeqLM` | LoRA | `model.generate()` |

Most modern models are CausalLM. Use `Adapt2BitNet/FT/finetune.py` as a template.

#### 3. Adapt the fine-tuning script

Key changes to make in `finetune.py`:

```python
def load_tokenizer_and_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    # Set pad token if missing
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model
```

Adjust `build_chat_input()` for the model's expected chat template format. Models using HuggingFace's `apply_chat_template()` work out of the box. Models with custom formats (like GLM-4's hardcoded token IDs) need manual construction.

#### 4. Configure LoRA targets

Identify the model's linear layer names:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("your-model")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(name)
```

Common patterns:

| Model Family | Target Modules |
|---|---|
| Llama / BitNet / Qwen | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| ChatGLM / GLM-4 | `query_key_value` (fused QKV) |
| Falcon | `query_key_value`, `dense`, `dense_h_to_4h`, `dense_4h_to_h` |
| Mistral | Same as Llama |

Write these into `configs/lora.yaml`:

```yaml
peft_config:
  peft_type: LORA
  task_type: CAUSAL_LM
  r: 8
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules: ["q_proj", "k_proj", ...]
```

#### 5. Write inference script

Follow the pattern in `Adapt2BitNet/inference_bitnet.py`:

1. Load base model
2. Stage 1: Load NLP LoRA adapter -> generate task identification -> unload
3. Stage 2: Load task LoRA adapter -> generate classification -> unload
4. Return result

#### 6. Register in config

Create `Adapt2<Name>/config.json`:

```json
{
    "model_path": "org/model-name",
    "adapter_path": "models/<name>/adapters/",
    "adapter_set": {
        "NLP": "instruction/",
        "MTD": "ustc-tfc-2016-detection-packet/"
    },
    "tasks": {
        "Malware Traffic Detection": "MTD",
        "Botnet Detection": "BND"
    }
}
```

---

## Adding a New Traffic Task

### Step-by-Step

#### 1. Preprocess the dataset

```bash
cd preprocess
python preprocess_dataset.py \
    --input /path/to/new/dataset \
    --dataset_name new-dataset-2024 \
    --traffic_task detection \
    --granularity packet \
    --output_path ../datasets/new-dataset-2024 \
    --output_name new-dataset-2024_detection_packet
```

If the dataset has a non-standard folder structure, add handling to `preprocess/specfic_dataset_utils.py`.

#### 2. Choose a task code

Pick a short uppercase abbreviation (e.g., `NDD` for "New Dataset Detection"). Check that it doesn't collide with existing codes in [architecture.md](architecture.md).

#### 3. Add a preprompt template

In the inference script for your backend, add a preprompt entry. For BitNet (`inference_bitnet.py`):

```python
def preprompt(task, traffic):
    prompts = {
        # ...existing tasks...
        "NDD": f"Given the following traffic data {traffic} Please conduct "
               f"the NEW DATASET DETECTION TASK to determine which category "
               f"the traffic belongs to.",
    }
    return prompts.get(task, f"Analyze: {traffic}")
```

For ChatGLM2 (`inference.py`), add the same mapping.

#### 4. Add to instruction dataset

Add instruction examples to `datasets/instructions/instructions.json`:

```json
{"instruction": "Please help me detect new dataset traffic.", "output": "New Dataset Detection"}
{"instruction": "I need to analyze packets from the new-2024 dataset.", "output": "New Dataset Detection"}
{"instruction": "Can you classify new-dataset-2024 traffic?", "output": "New Dataset Detection"}
```

Re-run Stage 1 training to update the NLP adapter.

#### 5. Train a task adapter

```bash
# ChatGLM2
cd dual-stage-tuning
bash trafficllm_stage2.sh  # edit paths in script

# BitNet
cd Adapt2BitNet/FT/scripts
bash trafficllm_stage2.sh new-dataset-2024-detection-packet \
    ../../datasets/new-dataset-2024/new-dataset-2024_detection_packet_train.json
```

#### 6. Register in config

Add entries to the appropriate `config.json`:

```json
{
    "adapter_set": {
        "NDD": "new-dataset-2024-detection-packet/"
    },
    "tasks": {
        "New Dataset Detection": "NDD"
    }
}
```

#### 7. Evaluate

```bash
python Adapt2BitNet/evaluation_bitnet.py \
    --model_name microsoft/bitnet-b1.58-2B-4T-bf16 \
    --traffic_task detection \
    --test_file datasets/new-dataset-2024/new-dataset-2024_detection_packet_test.json \
    --label_file datasets/new-dataset-2024/new-dataset-2024_label.json \
    --adapter_path models/bitnet/adapters/new-dataset-2024-detection-packet
```

---

## Project Conventions

### File Naming

| Type | Convention | Example |
|------|-----------|---------|
| Dataset files | `<dataset>_<task>_<granularity>_<split>.json` | `ustc-tfc-2016_detection_packet_train.json` |
| PEFT checkpoint dirs | `<dataset>-<task>-<granularity>/` | `ustc-tfc-2016-detection-packet/` |
| Shell scripts | `trafficllm_stage<N>.sh` | `trafficllm_stage1.sh` |
| Inference scripts | `inference.py` or `inference_<backend>.py` | `inference_bitnet.py` |
| Model adaptor dirs | `Adapt2<Name>/` | `Adapt2BitNet/` |

### Config Structure

Every backend config follows this schema:

```json
{
    "model_path": "<path or HF model ID>",
    "<adapter_key>": "<base path to adapters>",
    "<adapter_set_key>": {
        "<TASK_CODE>": "<relative adapter path>"
    },
    "tasks": {
        "<Human-Readable Task Name>": "<TASK_CODE>"
    }
}
```

ChatGLM2 uses `peft_path`/`peft_set`; BitNet uses `adapter_path`/`adapter_set`.

### PEFT Checkpoint Layout

**ChatGLM2 (P-Tuning v2):**
```
models/chatglm2/peft/<task>/checkpoint-<step>/
├── pytorch_model.bin          # Only prefix_encoder.* keys
├── config.json
├── tokenizer.model
└── tokenizer_config.json
```

**BitNet (LoRA):**
```
models/bitnet/adapters/<task>/
├── adapter_model.safetensors  # LoRA delta weights
├── adapter_config.json        # PEFT config (r, alpha, target_modules, ...)
└── README.md                  # Auto-generated by peft
```

### Dependencies and Version Constraints

| Backend | `transformers` | `peft` | `torch` | Notes |
|---------|---------------|--------|---------|-------|
| ChatGLM2 | 4.30.2 | N/A (uses built-in P-Tuning) | >= 2.0 | + `cpm_kernels` |
| Llama | >= 4.30 | from git | >= 2.0 | + `loralib` |
| GLM-4 | 4.44.0 | >= 0.7.0 | >= 2.0 | strict version lock |
| BitNet | >= 4.57 | >= 0.7.0 | >= 2.0 | + `accelerate`, `sentencepiece` |

These can conflict. Use separate conda environments for backends with incompatible `transformers` versions:

```bash
conda create -n trafficllm-chatglm python=3.9
pip install transformers==4.30.2

conda create -n trafficllm-bitnet python=3.10
pip install transformers>=4.57
```

---

## Testing and Validation Workflow

### Smoke Test After Training

Run a quick inference to verify the adapter loads and generates reasonable output:

```bash
# BitNet
python Adapt2BitNet/inference_bitnet.py \
    --config=Adapt2BitNet/config.json \
    --prompt="Please help me detect malware traffic.<packet>frame.encap_type: 1, frame.time: Mar 15 2024"
```

Expected behavior:
- Stage 1 should output a task name (e.g., "Malware Traffic Detection")
- Stage 2 should output a classification label (e.g., "Cridex")

### Evaluation Checklist

1. Run evaluation on the test split with `--max_samples 100` for a quick check
2. Verify accuracy is above random baseline (`1/num_classes`)
3. Check the confusion matrix for systematic misclassifications
4. Run with full test set for final metrics
5. Compare against the ChatGLM2 baseline on the same test set

### Common Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Empty output from `model.generate()` | Wrong `max_new_tokens` or truncated input | Increase `max_new_tokens`, check `max_input_length` |
| All predictions are the same class | Undertrained adapter | Increase `max_steps`, check learning rate |
| CUDA OOM during training | Batch size too large | Reduce `per_device_train_batch_size`, increase `gradient_accumulation_steps` |
| `KeyError: 'prefix_encoder'` | Wrong checkpoint for model type | Use P-Tuning checkpoint for ChatGLM2, LoRA for BitNet |
| `transformers` import error | Version mismatch | Check required version for your backend |
| Adapter loads but predictions are random | Wrong adapter or wrong base model | Verify paths in `config.json` |
