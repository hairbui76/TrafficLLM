# Inference and Deployment

## CLI Inference

### ChatGLM2

The primary inference script (`inference.py`) implements dual-stage inference using ChatGLM2's `.chat()` API.

```bash
python inference.py --config=config.json \
    --prompt="Please help me detect malware traffic.<packet>frame.encap_type: 1, frame.time: Mar 15 2024, tcp.payload: 2080008070..."
```

**How it works:**

1. Parses the prompt into instruction text and traffic data (split on `<packet>`)
2. Loads the base ChatGLM2 model with `pre_seq_len=128`
3. **Stage 1:** Loads the NLP prefix adapter, calls `model.chat(tokenizer, instruction_text)` to identify the task (e.g., "Malware Traffic Detection" -> "MTD")
4. **Stage 2:** Loads the task-specific prefix adapter, constructs a preprompt template, calls `model.chat(tokenizer, traffic_prompt)` to get the result

**Prompt format:**
```
<instruction text><packet><traffic data fields>
```

Example:
```
Please help me detect malware traffic.<packet>frame.encap_type: 1, frame.time: Mar 15 2024, ip.src: 10.0.0.1, tcp.payload: 2080008070...
```

### BitNet

The BitNet inference script (`Adapt2BitNet/inference_bitnet.py`) replaces `.chat()` with `model.generate()` and uses LoRA adapter hot-swap.

```bash
python Adapt2BitNet/inference_bitnet.py \
    --config=Adapt2BitNet/config.json \
    --prompt="Please help me detect malware traffic.<packet>frame.encap_type: 1, ..."
```

**How it works:**

1. Loads the base BitNet model via `AutoModelForCausalLM.from_pretrained()`
2. **Stage 1:** Loads NLP LoRA adapter via `PeftModel.from_pretrained()`, generates task identification using `model.generate()` with chat template
3. Unloads the NLP adapter via `model.unload()` to free memory
4. **Stage 2:** Loads the task-specific LoRA adapter, generates classification result
5. Uses `tokenizer.apply_chat_template()` for input formatting

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--config` | `config.json` | Path to config file |
| `--prompt` | (required) | Instruction + traffic data |
| `--device` | `cuda` | Device for inference |

---

## Streamlit Web Demo

The web demo provides a chat interface for TrafficLLM with optional PCAP file upload.

```bash
streamlit run trafficllm_server.py
```

Access at `http://localhost:8501` (or your server's IP).

**Features:**
- Text input for user instructions
- PCAP file upload (auto-extracts features via `tshark`)
- Adjustable generation parameters (max_length, top_p, temperature)
- Dual-stage inference with task identification and traffic analysis

**Configuration:** Edit `config.json` to point to your model and PEFT checkpoint paths.

**PCAP processing:** When a PCAP file is uploaded, the server runs `tshark` to extract 70+ protocol fields (frame, IP, TCP, UDP layers) and constructs the traffic data string automatically.

**Sidebar controls:**
- `max_length`: Maximum generation length (0-32768, default 8192)
- `top_p`: Nucleus sampling threshold (0.0-1.0, default 0.8)
- `temperature`: Sampling temperature (0.0-1.0, default 0.8)

---

## Agent System

TrafficLLM can be integrated into single-agent and multi-agent systems using [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent).

### Architecture

```
┌─────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│  Qwen3 LLM      │────>│  Qwen-Agent       │────>│  TrafficLLM      │
│  (Tool-calling)  │     │  (Orchestrator)   │     │  (Flask API)     │
│  vllm serve      │     │  assistant_qwen3  │     │  localhost:8877  │
└─────────────────┘     └───────────────────┘     └──────────────────┘
```

### Step 1: Deploy TrafficLLM as Flask API

```bash
cd agent
python trafficllm_flask.py
```

This starts an HTTP endpoint at `http://localhost:8877/api/conversation` that accepts JSON:

```json
{
    "human_instruction": "Please help me detect malware traffic.",
    "traffic_data": "<packet>frame.encap_type: 1, ..."
}
```

### Step 2: Deploy a Tool-Calling LLM

```bash
vllm serve Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --port 8000 \
    --max-model-len 10000 \
    --enable-reasoning \
    --reasoning-parser deepseek_r1
```

### Step 3: Run the Agent

**Single-agent:**
```bash
python assistant_qwen3.py
```

**Multi-agent (group chat):**
```bash
python group_chat.py
```

The multi-agent system creates specialized experts (malware analyst, VPN expert, etc.) that collaborate via a group chat protocol, each calling TrafficLLM as a tool for their domain.

### Tool Registration

TrafficLLM is registered as a Qwen-Agent tool:

```python
@register_tool('trafficllm')
class TrafficLLM(BaseTool):
    description = 'Network traffic detection service...'
    parameters = [{
        'name': 'prompt',
        'type': 'string',
        'description': 'Detailed description of the network traffic detection tasks and traffic data',
        'required': True,
    }]

    def call(self, params, **kwargs):
        prompt = json.loads(params)['prompt']
        human_instruction = prompt.split("<packet>")[0]
        traffic_data = "<packet>" + prompt.split("<packet>")[1]
        response = requests.post("http://localhost:8877/api/conversation",
                                 json={"human_instruction": human_instruction,
                                       "traffic_data": traffic_data})
        return response.json()
```

---

## Evaluation

### ChatGLM2 Evaluation

```bash
python evaluation.py \
    --model_name models/chatglm2/chatglm2-6b \
    --traffic_task detection \
    --test_file datasets/ustc-tfc-2016/ustc-tfc-2016_detection_packet_test.json \
    --label_file datasets/ustc-tfc-2016/ustc-tfc-2016_label.json \
    --ptuning_path models/chatglm2/peft/ustc-tfc-2016-detection-packet/checkpoint-20000/
```

### BitNet Evaluation

```bash
python Adapt2BitNet/evaluation_bitnet.py \
    --model_name microsoft/bitnet-b1.58-2B-4T-bf16 \
    --traffic_task detection \
    --test_file datasets/ustc-tfc-2016/ustc-tfc-2016_detection_packet_test.json \
    --label_file datasets/ustc-tfc-2016/ustc-tfc-2016_label.json \
    --adapter_path models/bitnet/adapters/ustc-tfc-2016-detection-packet \
    --max_samples 1000
```

### Evaluation Parameters

| Parameter | Description |
|-----------|-------------|
| `--model_name` | Base model path or HuggingFace ID |
| `--traffic_task` | `detection` or `generation` |
| `--test_file` | Path to test JSONL file |
| `--label_file` | Path to label mapping JSON (detection only) |
| `--ptuning_path` / `--adapter_path` | Path to PEFT checkpoint / LoRA adapter |
| `--max_samples` | Maximum test samples (default: 1000) |

### Metrics

**Detection tasks:** Accuracy, weighted precision, weighted recall, weighted F1, confusion matrix, per-class classification report.

**Generation tasks:** Saves generated outputs grouped by label to `generation.json` (ChatGLM2) or `generation_bitnet.json` (BitNet) for manual inspection or downstream evaluation.

---

## Traffic Generation Tutorial

TrafficLLM can generate synthetic network packets using Scapy.

### Setup

```bash
cd tutorials
mkdir -p tmp/packet_generation tmp/flow_generation
```

### Register Generation Tasks

Add generation checkpoints to `tutorials/config.json`:

```json
{
    "peft_set": {
        "MTG": ["ustc-tfc-2016-generation-packet-header/checkpoint-10000/",
                "ustc-tfc-2016-generation-packet-payload/checkpoint-10000/"]
    },
    "tasks": {
        "Malware Traffic Generation": "MTG"
    }
}
```

### Generate Packets

```bash
python generation.py --config=config.json \
    --prompt="Please generate a packet of Neris traffic."
```

The script:
1. Identifies the generation task via Stage 1
2. Generates packet headers (5-tuple) and hex payloads via Stage 2
3. Constructs packets using Scapy
4. Writes a pcap file (`synthetic_packet.pcap`) readable by Wireshark/tshark

### Verify Output

```bash
tshark -r synthetic_packet.pcap
```
