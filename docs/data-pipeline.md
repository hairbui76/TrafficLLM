# Data Pipeline

## Overview

TrafficLLM's data pipeline converts raw PCAP network captures into structured JSONL files suitable for LLM fine-tuning. The pipeline is model-agnostic -- the same preprocessed data can be used with any backend after optional format conversion.

```
Raw PCAP Files (labeled directories)
        │
        ▼
preprocess/preprocess_dataset.py
        │
        ├── packet_data_preprocess.py  (packet-level features via Scapy)
        ├── flow_data_preprocess.py    (flow-level features)
        └── specfic_dataset_utils.py   (dataset-specific logic)
        │
        ▼
JSONL Output: {"instruction": "...", "output": "..."}
  + Label File: {"label_name": index, ...}
  + Train/Test Split
        │
        ▼ (optional, for BitNet/GLM-4)
Adapt2BitNet/data/convert_to_chat.py
        │
        ▼
Chat JSONL: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
```

## Preprocessing CLI

The main entry point is `preprocess/preprocess_dataset.py`:

```bash
cd preprocess
python preprocess_dataset.py \
    --input /path/to/raw/dataset \
    --dataset_name ustc-tfc-2016 \
    --traffic_task detection \
    --granularity packet \
    --output_path /path/to/output \
    --output_name ustc-tfc-2016
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `--input` | str | Path to the raw traffic dataset directory. Must contain labeled sub-directories, each with `.pcap` files. |
| `--dataset_name` | str | Name of the dataset. Must match a registered name in the code (e.g., `ustc-tfc-2016`, `iscx-botnet`, `iscx-vpn-2016`). |
| `--traffic_task` | str | One of `detection`, `generation`, or `understanding`. |
| `--granularity` | str | One of `packet` (per-packet features) or `flow` (per-flow features). |
| `--output_path` | str | Directory for output files. |
| `--output_name` | str | Base name for output files. |

### Registered Dataset Names

The preprocessing script recognizes these dataset names and routes them to the appropriate detection task code:

| `--dataset_name` | Detection Task | Code |
|---|---|---|
| `ustc-tfc-2016` | Encrypted Malware Detection | EMD |
| `iscx-botnet` | Botnet Detection | BND |
| `iscx-vpn-2016` | Encrypted VPN Detection | EVD |
| `lfett-2021` | Encrypted VPN Detection | EVD |
| `dohbrw-2020` | Malicious DoH Detection | MDD |
| `iscx-tor-2016` | Tor Behavior Detection | TBD |
| `dapt-2020` | APT Attack Detection | APT |
| (other) | Encrypted App Classification | EAC |

## Internal Processing Flow

### 1. `preprocess_dataset.py`

Routes to the appropriate preprocessor based on `--traffic_task`:
- `detection` → `traffic_detection_preprocess()` -- builds labeled classification data
- `generation` → `traffic_generation_preprocess()` -- builds traffic generation data (hex sequences)
- `understanding` → `traffic_understanding_preprocess()` -- builds traffic understanding data

### 2. `preprocess_utils.py`

Core utilities:
- `build_dataset(args, input_path, label)` -- reads PCAPs from a labeled directory, splits into train/test
- `build_td_text_dataset()` -- converts raw packet/flow data into detection task text with instruction templates
- `build_tg_text_dataset()` -- converts data into generation task text
- `build_tu_text_dataset()` -- converts data into understanding task text
- `save_dataset()` -- writes train/test JSONL files
- `write_labels()` -- writes label-to-index mapping JSON

### 3. `packet_data_preprocess.py`

Uses Scapy to extract packet-level features:
- Protocol fields (frame, IP, TCP/UDP headers)
- 5-tuple information (src/dst IP, src/dst port, protocol)
- Hex payload data
- Feature strings for LLM consumption

### 4. `flow_data_preprocess.py`

Extracts flow-level features:
- Concatenated packet hex within a flow
- Flow sequence features via `flowcontainer`
- Aggregated statistics

### 5. `specfic_dataset_utils.py`

Dataset-specific preprocessing logic, e.g., `ustc_tfc2016_preprocess()` handles the USTC-TFC-2016 folder structure with Benign/Malware subdirectories.

## Output Formats

### Standard JSONL (ChatGLM2, Llama)

One JSON object per line with `instruction` and `output` fields:

```json
{"instruction": "Given the following traffic data <packet>frame.encap_type: 1, frame.time: ..., tcp.payload: 2080008070... Please conduct the ENCRYPTED MALWARE DETECTION TASK to determine which application category the encrypted traffic belongs to.", "output": "Cridex"}
```

### Chat Messages JSONL (BitNet, GLM-4)

Converted via `Adapt2BitNet/data/convert_to_chat.py`:

```json
{"messages": [{"role": "user", "content": "Given the following traffic data <packet>frame.encap_type: 1, ..."}, {"role": "assistant", "content": "Cridex"}]}
```

### Label File

JSON mapping label names to integer indices:

```json
{"BitTorrent": 0, "Cridex": 1, "FTP": 2, "Gmail": 3, ...}
```

## Chat Format Conversion

For model backends that use chat templates (BitNet, GLM-4), convert the standard JSONL:

```bash
python Adapt2BitNet/data/convert_to_chat.py \
    --input_file datasets/ustc-tfc-2016/ustc-tfc-2016_detection_packet_train.json \
    --output_file datasets/ustc-tfc-2016_chat/train.jsonl
```

The converter handles both JSON array files and JSONL files, and preserves conversation history if present.

## Traffic-Domain Tokenizer

TrafficLLM optionally trains a custom SentencePiece BPE tokenizer on traffic data to reduce token counts for traffic-specific text.

```bash
cd tokenization
python traffic_tokenizer.py
```

Configuration (edit in `traffic_tokenizer.py`):
- `model_name` -- path to the base model containing the native tokenizer
- `data_path` -- path to training datasets from the preprocessing step

The script trains a 64,794-token vocabulary and provides `tokenizer_comparing()` to benchmark token lengths against the base model's tokenizer.

## Supported Datasets

| Dataset | Year | Task | Granularity | Samples |
|---------|------|------|-------------|---------|
| USTC-TFC-2016 | 2016 | Malware Detection | Packet/Flow | 50.7K |
| ISCX-Botnet-2014 | 2014 | Botnet Detection | Packet/Flow | 25.0K |
| DoHBrw-2020 | 2020 | DoH Detection | Packet/Flow | 47.8K |
| CSIC-2010 | 2010 | Web Attack Detection | Packet | 34.5K |
| DAPT-2020 | 2020 | APT Detection | Packet | 10.0K |
| ISCX-VPN-2016 | 2016 | VPN Detection | Packet/Flow | 64.8K |
| ISCX-Tor-2016 | 2016 | Tor Detection | Packet/Flow | 40.0K |
| CSTNET-2023 | 2023 | App Classification | Packet/Flow | 97.6K |
| CW-100-2018 | 2018 | Website Fingerprinting | Flow | 7.4K |
| APP-53-2023 | 2023 | Concept Drift | Packet/Flow | 109.8K |

All datasets are available via the [Google Drive link](https://drive.google.com/drive/folders/1RZAOPcNKq73-quA8KG_lkAo_EqlwhlQb).
