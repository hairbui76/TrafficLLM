# EA-PEFT: Extensible Adaptation with Parameter-Efficient Fine-Tuning

## Overview

EA-PEFT is TrafficLLM's mechanism for organizing and managing multiple task-specific adapters with minimal overhead. It enables two key operations:

- **Update:** Add a new task adapter to the model's repertoire
- **Register:** Re-train an existing task adapter with new data (e.g., when traffic patterns change)

This is critical for real-world deployment where network traffic evolves and new attack types emerge. Instead of retraining the entire model, EA-PEFT fine-tunes only a small set of adapter parameters per task.

## How It Works

```
                    ┌──────────────────────────┐
                    │       EA-PEFT CLI         │
                    │    EA-PEFT/ea-peft.py     │
                    └────────┬─────────────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌──────────────┐  ┌────────────┐  ┌────────────────┐
    │  Stage 1     │  │  Update    │  │  Register      │
    │  Instruction │  │  (new task │  │  (retrain      │
    │  Tuning      │  │  adapter)  │  │  existing)     │
    └──────────────┘  └────────────┘  └────────────────┘
            │                │                │
            ▼                ▼                ▼
    ┌──────────────────────────────────────────────────┐
    │              models/chatglm2/peft/               │
    │  ├── instruction/checkpoint-8000/                │
    │  ├── ustc-tfc-2016-detection-packet/             │
    │  ├── iscx-botnet-2014-detection-packet/          │
    │  ├── csic-2010-detection-packet/                 │
    │  └── ... (one directory per task)                │
    └──────────────────────────────────────────────────┘
```

## CLI Usage

```bash
cd EA-PEFT
python ea-peft.py \
    --model_name /path/to/chatglm2-6b \
    --tuning_data /path/to/new/data \
    --adaptation_task update \
    --task_name MTD
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--model_name` | Path to the base model (ChatGLM2) |
| `--tuning_data` | Directory containing `instructions/instruction.json` and `traffic/traffic.json` |
| `--adaptation_task` | `update` (add new task) or `register` (retrain existing task) |
| `--task_name` | Downstream task code (e.g., MTD, BND, EVD) |

## Internal Workflow

### 1. Stage 1 Tuning (Always Runs)

Regardless of the adaptation task, EA-PEFT first runs Stage 1 instruction tuning to ensure the NLP adapter is up-to-date:

```python
def stage1_tuning(model_name, instruction_data):
    cmd = "torchrun ... main.py --train_file " + instruction_data + \
          " --output_dir ../models/chatglm2/peft/instruction --pre_seq_len 128"
    os.system(cmd)
```

This re-trains the instruction understanding adapter using `dual-stage-tuning/main.py`.

### 2. Update Operation (New Task)

Adds a new task that does not yet exist:

```python
def model_update(model_name, traffic_data, task_name):
    assert task_name not in os.listdir("../models/chatglm2/peft/")
    stage2_tuning(model_name, traffic_data, task_name)
```

- Asserts the task directory does NOT already exist
- Runs Stage 2 tuning to create the new adapter under `models/chatglm2/peft/<task_name>/`

### 3. Register Operation (Retrain Existing)

Updates an existing task with new training data:

```python
def model_insert(model_name, traffic_data, task_name):
    assert task_name in os.listdir("../models/chatglm2/peft/")
    os.mkdir(os.path.join("../models/chatglm2/peft/", task_name))
    stage2_tuning(model_name, traffic_data, task_name)
```

- Asserts the task directory DOES already exist
- Creates a subdirectory and trains a new checkpoint

### 4. Stage 2 Tuning

Both operations use the same Stage 2 tuning function:

```python
def stage2_tuning(model_name, traffic_data, task_name):
    cmd = "torchrun ... main.py --train_file " + traffic_data + \
          " --output_dir ../models/chatglm2/peft/" + task_name + " --pre_seq_len 128"
    os.system(cmd)
```

## Input Data Format

The `--tuning_data` directory must contain:

```
tuning_data/
├── instructions/
│   └── instruction.json    # Task understanding data
└── traffic/
    └── traffic.json        # Task-specific traffic data
```

Both files use the standard JSONL format with `instruction` and `output` fields.

## Adapter Directory Structure

After training, adapters are stored under `models/chatglm2/peft/`:

```
models/chatglm2/peft/
├── instruction/
│   └── checkpoint-8000/
│       ├── pytorch_model.bin        # prefix_encoder weights
│       ├── config.json              # model config with pre_seq_len
│       ├── tokenizer_config.json
│       └── ...
├── ustc-tfc-2016-detection-packet/
│   └── checkpoint-10000/
│       └── ...
├── iscx-botnet-2014-detection-packet/
│   └── checkpoint-5000/
│       └── ...
└── ...
```

Each checkpoint contains:
- `pytorch_model.bin` -- only the `transformer.prefix_encoder.*` weights (P-Tuning v2)
- Copies of the tokenizer and config files

## Extending EA-PEFT for BitNet

The EA-PEFT concept extends naturally to the BitNet backend, though the mechanism differs:

| Aspect | ChatGLM2 EA-PEFT | BitNet Equivalent |
|--------|-------------------|-------------------|
| Storage | `models/chatglm2/peft/<task>/checkpoint-*` | `models/bitnet/adapters/<task>/` |
| Adapter type | Prefix encoder weights | LoRA adapter files |
| Loading | Manual `prefix_state_dict` swap | `PeftModel.from_pretrained()` |
| Training | `dual-stage-tuning/main.py` | `Adapt2BitNet/FT/finetune.py` |

To add a new task for BitNet:

1. Preprocess the new dataset
2. Convert to chat format: `python Adapt2BitNet/data/convert_to_chat.py --input_file ... --output_file ...`
3. Run Stage 2: `bash Adapt2BitNet/FT/scripts/trafficllm_stage2.sh <task_name> <train_file>`
4. Register the adapter path in `Adapt2BitNet/config.json`
5. Add the preprompt template in `Adapt2BitNet/inference_bitnet.py`

## Design Rationale

EA-PEFT addresses the fundamental challenge of **concept drift** in network traffic:

- Attack tools evolve, generating new traffic patterns
- New applications create new encrypted traffic signatures
- Network configurations change, shifting baseline behavior

By isolating each task into a small adapter (~few MB for prefix weights, ~34MB for LoRA adapters), TrafficLLM can:
- Update individual tasks without affecting others
- Add new tasks without retraining the base model
- Maintain a library of adapters for different network environments
- Roll back to previous adapters if new training degrades performance
