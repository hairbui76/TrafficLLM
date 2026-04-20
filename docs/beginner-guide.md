# TrafficLLM - Hướng dẫn toàn diện cho người mới bắt đầu

> Tài liệu này giải thích **toàn bộ** những gì bạn cần biết để hiểu repo TrafficLLM, từ kiến thức nền tảng AI đến chi tiết từng dòng code.

---

## Mục lục

- [TrafficLLM - Hướng dẫn toàn diện cho người mới bắt đầu](#trafficllm---hướng-dẫn-toàn-diện-cho-người-mới-bắt-đầu)
  - [Mục lục](#mục-lục)
  - [1. Kiến thức nền tảng AI cần biết trước](#1-kiến-thức-nền-tảng-ai-cần-biết-trước)
    - [1.1 Large Language Model (LLM) là gì?](#11-large-language-model-llm-là-gì)
    - [1.2 Tokenization là gì?](#12-tokenization-là-gì)
    - [1.3 Fine-tuning là gì?](#13-fine-tuning-là-gì)
    - [1.4 PEFT và P-Tuning v2 là gì?](#14-peft-và-p-tuning-v2-là-gì)
    - [1.5 LoRA là gì?](#15-lora-là-gì)
    - [1.6 Inference là gì?](#16-inference-là-gì)
    - [1.7 Evaluation metrics: Accuracy, Precision, Recall, F1](#17-evaluation-metrics-accuracy-precision-recall-f1)
  - [2. Kiến thức về Network Traffic](#2-kiến-thức-về-network-traffic)
    - [2.1 Network Traffic là gì?](#21-network-traffic-là-gì)
    - [2.2 PCAP file là gì?](#22-pcap-file-là-gì)
    - [2.3 Tshark và Wireshark](#23-tshark-và-wireshark)
    - [2.4 Packet vs Flow](#24-packet-vs-flow)
    - [2.5 Các bài toán phân tích traffic](#25-các-bài-toán-phân-tích-traffic)
  - [3. TrafficLLM là gì? Tổng quan kiến trúc](#3-trafficllm-là-gì-tổng-quan-kiến-trúc)
    - [3.1 Ý tưởng cốt lõi](#31-ý-tưởng-cốt-lõi)
    - [3.2 Ba kỹ thuật chính](#32-ba-kỹ-thuật-chính)
    - [3.3 Dual-Stage Pipeline (Quy trình 2 giai đoạn)](#33-dual-stage-pipeline-quy-trình-2-giai-đoạn)
  - [4. Cấu trúc thư mục repo](#4-cấu-trúc-thư-mục-repo)
  - [5. Chi tiết từng module](#5-chi-tiết-từng-module)
    - [5.1 File gốc (root)](#51-file-gốc-root)
      - [`config.json` — Bản đồ cấu hình trung tâm](#configjson--bản-đồ-cấu-hình-trung-tâm)
      - [`inference.py` — Inference dòng lệnh](#inferencepy--inference-dòng-lệnh)
      - [`evaluation.py` — Đánh giá model](#evaluationpy--đánh-giá-model)
      - [`trafficllm_server.py` — Web demo (upload PCAP)](#trafficllm_serverpy--web-demo-upload-pcap)
      - [`trafficllm_server_text.py` — Web demo (nhập text)](#trafficllm_server_textpy--web-demo-nhập-text)
    - [5.2 `preprocess/` — Tiền xử lý dữ liệu](#52-preprocess--tiền-xử-lý-dữ-liệu)
    - [5.3 `tokenization/` — Tokenizer chuyên biệt](#53-tokenization--tokenizer-chuyên-biệt)
    - [5.4 `dual-stage-tuning/` — Training pipeline](#54-dual-stage-tuning--training-pipeline)
    - [5.5 `EA-PEFT/` — Mở rộng và cập nhật model](#55-ea-peft--mở-rộng-và-cập-nhật-model)
    - [5.6 `datasets/` — Dữ liệu huấn luyện](#56-datasets--dữ-liệu-huấn-luyện)
    - [5.7 `models/` — Checkpoints đã train](#57-models--checkpoints-đã-train)
    - [5.8 `tutorials/` — Tạo traffic với Scapy](#58-tutorials--tạo-traffic-với-scapy)
    - [5.9 `Adapt2BitNet/` — Backbone BitNet 1.58-bit](#59-adapt2bitnet--backbone-bitnet-158-bit)
    - [5.10 `Adapt2GLM4/` — Backbone GLM4](#510-adapt2glm4--backbone-glm4)
    - [5.11 `llm/` — Backbone Llama2 \& DeepSeek-R1](#511-llm--backbone-llama2--deepseek-r1)
    - [5.12 `agent/` — Hệ thống Agent](#512-agent--hệ-thống-agent)
  - [6. Luồng dữ liệu end-to-end](#6-luồng-dữ-liệu-end-to-end)
  - [7. Các downstream task cụ thể](#7-các-downstream-task-cụ-thể)
    - [Traffic Detection Tasks (10 tasks)](#traffic-detection-tasks-10-tasks)
    - [Traffic Generation Tasks (4 tasks)](#traffic-generation-tasks-4-tasks)
  - [8. Datasets chi tiết](#8-datasets-chi-tiết)
    - [Instruction Dataset](#instruction-dataset)
    - [Traffic Datasets](#traffic-datasets)
  - [9. Config.json — Bản đồ cấu hình](#9-configjson--bản-đồ-cấu-hình)
    - [ChatGLM2 config (root `config.json`)](#chatglm2-config-root-configjson)
    - [BitNet config (`Adapt2BitNet/config.json`)](#bitnet-config-adapt2bitnetconfigjson)
  - [10. Cách chạy từng phần](#10-cách-chạy-từng-phần)
    - [10.1 Cài đặt môi trường](#101-cài-đặt-môi-trường)
    - [10.2 Download model](#102-download-model)
    - [10.3 Preprocessing raw traffic](#103-preprocessing-raw-traffic)
    - [10.4 Training](#104-training)
    - [10.5 Evaluation](#105-evaluation)
    - [10.6 Inference (Terminal)](#106-inference-terminal)
    - [10.7 Web Demo](#107-web-demo)
    - [10.8 Traffic Generation](#108-traffic-generation)
  - [11. Yêu cầu phần cứng](#11-yêu-cầu-phần-cứng)
  - [12. Thuật ngữ (Glossary)](#12-thuật-ngữ-glossary)
13. [Phụ lục: Các khái niệm thuật toán & kỹ thuật Deep Learning](#13-phụ-lục-các-khái-niệm-thuật-toán--kỹ-thuật-deep-learning)
    - 13.1 [Transformer Architecture](#131-transformer-architecture)
    - 13.2 [Self-Attention Mechanism](#132-self-attention-mechanism)
    - 13.3 [Multi-Head Attention & Grouped Query Attention](#133-multi-head-attention--grouped-query-attention)
    - 13.4 [Positional Encoding / Embedding](#134-positional-encoding--embedding)
    - 13.5 [Feed-Forward Network (FFN) & Gate Mechanism](#135-feed-forward-network-ffn--gate-mechanism)
    - 13.6 [Causal LM vs Seq2Seq (Encoder-Decoder)](#136-causal-lm-vs-seq2seq-encoder-decoder)
    - 13.7 [Loss Function: Cross-Entropy](#137-loss-function-cross-entropy)
    - 13.8 [Backpropagation & Gradient Descent](#138-backpropagation--gradient-descent)
    - 13.9 [Learning Rate, Warmup & Scheduler](#139-learning-rate-warmup--scheduler)
    - 13.10 [Gradient Accumulation](#1310-gradient-accumulation)
    - 13.11 [Gradient Checkpointing](#1311-gradient-checkpointing)
    - 13.12 [Mixed Precision Training (FP16 / BF16)](#1312-mixed-precision-training-fp16--bf16)
    - 13.13 [BPE (Byte Pair Encoding) Tokenization](#1313-bpe-byte-pair-encoding-tokenization)
    - 13.14 [Prefix Tuning / P-Tuning v2 (thuật toán chi tiết)](#1314-prefix-tuning--p-tuning-v2-thuật-toán-chi-tiết)
    - 13.15 [LoRA — Low-Rank Adaptation (thuật toán chi tiết)](#1315-lora--low-rank-adaptation-thuật-toán-chi-tiết)
    - 13.16 [Text Generation: Temperature, Top-p, Top-k, Beam Search](#1316-text-generation-temperature-top-p-top-k-beam-search)
    - 13.17 [Chat Template & Special Tokens](#1317-chat-template--special-tokens)
    - 13.18 [Quantization & Ternary Weights (BitNet)](#1318-quantization--ternary-weights-bitnet)
    - 13.19 [Distributed Training (DDP, torchrun)](#1319-distributed-training-ddp-torchrun)
    - 13.20 [Evaluation: ROUGE & BLEU (cho Generation)](#1320-evaluation-rouge--bleu-cho-generation)
    - 13.21 [Confusion Matrix & Classification Report](#1321-confusion-matrix--classification-report)

---

## 1. Kiến thức nền tảng AI cần biết trước

### 1.1 Large Language Model (LLM) là gì?

LLM (Mô hình ngôn ngữ lớn) là một chương trình AI được huấn luyện trên lượng lớn văn bản để **hiểu và sinh ra ngôn ngữ tự nhiên**. Ví dụ: ChatGPT, ChatGLM, Llama.

**Cách hoạt động đơn giản:**
```
Input: "Thủ đô của Việt Nam là"
  → LLM xử lý →
Output: "Hà Nội"
```

LLM nhận input là chuỗi text (gọi là **prompt**), xử lý qua hàng tỷ tham số (parameters), và sinh ra output text.

**Trong TrafficLLM**, LLM được sử dụng để:
- Nhận input là **dữ liệu network traffic** (thay vì text thông thường)
- Output là **kết quả phân tích** (ví dụ: "đây là malware loại Zeus")

### 1.2 Tokenization là gì?

Máy tính không hiểu chữ — nó chỉ hiểu số. **Tokenization** là quá trình chuyển text thành dãy số (tokens) mà model có thể xử lý.

```
"Hello world" → [15496, 995]        (mỗi từ/phần từ được gán 1 số ID)
```

**Vấn đề với traffic data:** Tokenizer của ChatGLM được train trên text tiếng Trung/Anh, nên khi gặp dữ liệu traffic như `tcp.payload: 47:45:54:20:2f`, nó sẽ tách rất kém hiệu quả (tốn nhiều token hơn cần thiết).

**Giải pháp của TrafficLLM:** Train một **traffic-domain tokenizer** chuyên biệt sử dụng SentencePiece BPE để xử lý dữ liệu traffic hiệu quả hơn (ít token hơn = xử lý nhanh hơn).

### 1.3 Fine-tuning là gì?

**Fine-tuning** là quá trình "dạy thêm" cho một model đã được huấn luyện sẵn (pre-trained) để nó giỏi hơn ở một nhiệm vụ cụ thể.

```
Pre-trained LLM (biết ngôn ngữ chung)
    ↓ Fine-tune với traffic data
TrafficLLM (giỏi phân tích network traffic)
```

Giống như một bác sĩ đã học y khoa tổng quát, sau đó chuyên khoa thêm về tim mạch.

### 1.4 PEFT và P-Tuning v2 là gì?

**PEFT** (Parameter-Efficient Fine-Tuning) là kỹ thuật fine-tune mà **chỉ cập nhật một phần nhỏ tham số** thay vì toàn bộ model. Điều này:
- Tiết kiệm bộ nhớ GPU (từ 48GB xuống 8-16GB)
- Train nhanh hơn
- Có thể lưu nhiều "kỹ năng" khác nhau dưới dạng các file nhỏ

**P-Tuning v2** (dùng trong ChatGLM2 path):
- Thêm một lớp "prefix" nhỏ vào đầu model
- Chỉ train lớp prefix này (~2MB), giữ nguyên model gốc (~12GB)
- Mỗi task có 1 file `pytorch_model.bin` nhỏ riêng

```
Model gốc ChatGLM2 (12GB, không thay đổi)
    + Prefix Encoder cho task MTD (2MB)
    + Prefix Encoder cho task BND (2MB)
    + Prefix Encoder cho task EVD (2MB)
    ...
```

### 1.5 LoRA là gì?

**LoRA** (Low-Rank Adaptation) là một kỹ thuật PEFT phổ biến hơn, được dùng trong **Adapt2BitNet** và **Adapt2GLM4**.

Thay vì thay đổi trọng số gốc của model, LoRA thêm **các ma trận nhỏ** bên cạnh các lớp linear có sẵn:

```
Lớp gốc:    y = Wx            (W rất lớn, đông cứng)
Với LoRA:    y = Wx + BAx      (B, A là ma trận nhỏ, chỉ train B và A)
```

Tham số quan trọng:
- `r` (rank): kích thước ma trận nhỏ (r=8 hay r=16). Lớn hơn = mạnh hơn nhưng tốn hơn
- `lora_alpha`: hệ số scale
- `target_modules`: lớp nào trong model sẽ được gắn LoRA

### 1.6 Inference là gì?

**Inference** là quá trình **sử dụng model đã train để dự đoán** trên dữ liệu mới. Trong TrafficLLM, inference có 2 giai đoạn:

```
Giai đoạn 1 (Task Understanding):
  Input: "Please help me detect malware traffic"
  → NLP adapter xử lý →
  Output: "Malware Traffic Detection"

Giai đoạn 2 (Traffic Analysis):
  Input: preprompt + traffic data (packet fields, payload...)
  → MTD adapter xử lý →
  Output: "Zeus" (loại malware)
```

### 1.7 Evaluation metrics: Accuracy, Precision, Recall, F1

Khi đánh giá model, ta dùng các chỉ số:

| Metric | Ý nghĩa | Công thức |
|--------|---------|-----------|
| **Accuracy** | Tỷ lệ dự đoán đúng tổng thể | Đúng / Tổng |
| **Precision** | Trong những cái model nói "có", bao nhiêu thực sự "có" | TP / (TP + FP) |
| **Recall** | Trong những cái thực sự "có", model tìm được bao nhiêu | TP / (TP + FN) |
| **F1-Score** | Trung bình điều hòa của Precision và Recall | 2 × P × R / (P + R) |

Ví dụ: Model nói "đây là malware Zeus" 100 lần. Trong đó 90 lần đúng (TP=90), 10 lần sai (FP=10). Precision = 90%.

---

## 2. Kiến thức về Network Traffic

### 2.1 Network Traffic là gì?

Network traffic là **dữ liệu truyền qua mạng** giữa các thiết bị. Mỗi khi bạn truy cập web, gửi email, hay download file, đều tạo ra traffic.

Traffic bao gồm:
- **Header**: thông tin địa chỉ (IP nguồn/đích, port, protocol...)
- **Payload**: nội dung thực sự được truyền (có thể bị mã hóa)

### 2.2 PCAP file là gì?

PCAP (Packet Capture) là **file ghi lại toàn bộ network traffic** dưới dạng nhị phân. Nó chứa tất cả các packet đã đi qua một điểm mạng.

```
test.pcap → chứa hàng trăm/ngàn packets, mỗi packet có đầy đủ header + payload
```

### 2.3 Tshark và Wireshark

- **Wireshark**: Ứng dụng GUI để xem và phân tích PCAP files
- **Tshark**: Phiên bản command-line của Wireshark

TrafficLLM dùng `tshark` để **trích xuất các trường** từ PCAP file thành dạng text:

```bash
tshark -r test.pcap -e frame.time -e ip.src -e tcp.dstport -e tcp.payload ...
```

Output:
```
frame.encap_type: 1, frame.time: Aug 19, 2011, ip.src: 147.32.84.191, tcp.dstport: 80, tcp.payload: 47:45:54:20:2f...
```

### 2.4 Packet vs Flow

| Khái niệm | Giải thích |
|-----------|-----------|
| **Packet** | Một gói tin đơn lẻ truyền qua mạng. Đây là đơn vị nhỏ nhất. |
| **Flow** | Một chuỗi packets thuộc cùng một kết nối (cùng src IP, dst IP, src port, dst port, protocol). Ví dụ: toàn bộ packets của 1 phiên HTTP. |

TrafficLLM hỗ trợ cả **packet-level** và **flow-level** analysis.

### 2.5 Các bài toán phân tích traffic

| Loại | Bài toán | Mô tả |
|------|---------|-------|
| **Detection** | Phát hiện | Phân loại traffic thuộc loại nào (malware? botnet? VPN?) |
| **Generation** | Sinh dữ liệu | Tạo ra traffic giả lập giống traffic thật (dùng cho testing) |
| **Understanding** | Hiểu | Giải thích ý nghĩa các trường trong protocol |

---

## 3. TrafficLLM là gì? Tổng quan kiến trúc

### 3.1 Ý tưởng cốt lõi

TrafficLLM là một **framework thích ứng LLM cho phân tích network traffic**. Thay vì train model từ đầu, nó lấy một LLM có sẵn (ChatGLM2, GLM4, BitNet, Llama...) và "dạy thêm" để hiểu traffic data.

**Paper**: [arXiv:2504.04222](https://arxiv.org/abs/2504.04222)

### 3.2 Ba kỹ thuật chính

```
┌────────────────────────────────────────────────────────────────┐
│                      TrafficLLM Framework                      │
│                                                                │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────┐ │
│  │  1. Traffic  │  │ 2. Dual-Stage    │  │  3. EA-PEFT       │ │
│  │  Domain      │  │    Tuning        │  │  (Extensible      │ │
│  │ Tokenization │  │    Pipeline      │  │   Adaptation)     │ │
│  │              │  │                  │  │                   │ │
│  │ Chuyển đổi   │  │ Stage 1: Hiểu    │  │ Cập nhật model    │ │
│  │ traffic data │  │ instruction      │  │ cho môi trường    │ │
│  │ thành tokens │  │ Stage 2: Phân    │  │ mới mà không      │ │
│  │ hiệu quả     │  │ tích traffic     │  │ train lại toàn bộ │ │
│  └──────────────┘  └──────────────────┘  └───────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

### 3.3 Dual-Stage Pipeline (Quy trình 2 giai đoạn)

Đây là thiết kế cốt lõi của TrafficLLM:

```
INPUT: Người dùng gửi instruction + traffic data
           │
           ▼
┌─────────────────────────────────┐
│  STAGE 1: Task Understanding    │
│  (NLP Adapter)                  │
│                                 │
│  Input: "Please detect if this  │
│   traffic is malware"           │
│  Output: "Malware Traffic       │
│   Detection"                    │
│                                 │
│  → Xác định user muốn làm gì    │
└──────────────┬──────────────────┘
               │ task = "MTD"
               ▼
┌─────────────────────────────────┐
│  STAGE 2: Traffic Analysis      │
│  (Task-Specific Adapter)        │
│                                 │
│  Input: preprompt + packet data │
│   (frame.time, ip.src, tcp.     │
│    payload...)                  │
│  Output: "Zeus"                 │
│                                 │
│  → Phân tích traffic cụ thể     │
└──────────────┬──────────────────┘
               │
               ▼
OUTPUT: "Downstream task: Malware Traffic Detection"
        "Predicted result: Zeus"
```

**Tại sao cần 2 giai đoạn?**
- Stage 1 cho phép hệ thống **tự động chọn đúng adapter** dựa trên câu hỏi của user
- Stage 2 dùng adapter **chuyên biệt** cho task đó → chính xác hơn so với 1 model làm tất cả

---

## 4. Cấu trúc thư mục repo

```
TrafficLLM/
│
├── config.json                    # Cấu hình model path + PEFT adapters
├── requirements.txt               # Dependencies Python
├── inference.py                   # Inference terminal mode (ChatGLM2)
├── evaluation.py                  # Đánh giá model (ChatGLM2)
├── trafficllm_server.py           # Web demo Streamlit (PCAP upload)
├── trafficllm_server_text.py      # Web demo Streamlit (text input)
│
├── preprocess/                    # Tiền xử lý raw PCAP → JSON dataset
│   ├── preprocess_dataset.py      # Entry point
│   ├── packet_data_preprocess.py  # Trích xuất packet features
│   ├── flow_data_preprocess.py    # Trích xuất flow features
│   ├── preprocess_utils.py        # Utility functions
│   └── specfic_dataset_utils.py   # Dataset-specific handlers
│
├── tokenization/                  # Traffic-domain tokenizer
│   └── traffic_tokenizer.py       # Train BPE tokenizer cho traffic
│
├── dual-stage-tuning/             # Training pipeline (ChatGLM2)
│   ├── main.py                    # Training entry point
│   ├── arguments.py               # Cấu hình training arguments
│   ├── trainer.py                 # Custom Trainer class
│   ├── trainer_seq2seq.py         # Seq2Seq Trainer override
│   ├── trafficllm_stage1.sh       # Script Stage 1 (instruction)
│   └── trafficllm_stage2.sh       # Script Stage 2 (traffic)
│
├── EA-PEFT/                       # Extensible Adaptation
│   └── ea-peft.py                 # Update/register new tasks
│
├── datasets/                      # Datasets đã xử lý (JSON)
│   ├── instructions/              # 9,209 human instructions
│   │   └── instruction.json
│   ├── ustc-tfc-2016/             # Malware Traffic Detection
│   ├── iscx-botnet-2014/          # Botnet Detection
│   ├── csic-2010/                 # Web Attack Detection
│   ├── dapt-2020/                 # APT Attack Detection
│   ├── iscx-vpn-2016/             # Encrypted VPN Detection
│   ├── iscx-tor-2016/             # Tor Behavior Detection
│   ├── cstnet-2023/               # Encrypted App Classification
│   ├── dohbrw-2020/               # Malicious DoH Detection
│   ├── cw100-2018-2024/           # Website Fingerprinting
│   └── app53-2023/                # Concept Drift
│
├── models/                        # Pre-trained + fine-tuned models
│   └── chatglm2/
│       └── peft/                  # PEFT checkpoints cho mỗi task
│           ├── instruction/       # Stage 1 adapter (NLP)
│           ├── ustc-tfc-2016-.../  # Stage 2 adapter (MTD)
│           ├── iscx-botnet-.../    # Stage 2 adapter (BND)
│           └── ...
│
├── Adapt2BitNet/                  # BitNet b1.58-2B backbone
│   ├── inference_bitnet.py        # Inference cho BitNet
│   ├── evaluation_bitnet.py       # Evaluation cho BitNet
│   ├── config.json                # Config cho BitNet
│   ├── FT/                        # Fine-tuning scripts
│   │   ├── finetune.py            # LoRA fine-tuning
│   │   ├── configs/lora.yaml      # Hyperparameters
│   │   └── scripts/               # Shell scripts
│   └── data/
│       └── convert_to_chat.py     # Chuyển format data
│
├── Adapt2GLM4/                    # GLM4-9B backbone
│   ├── Preprocess.py              # Chuyển format cho GLM4
│   └── FT/                        # Fine-tuning scripts
│
├── llm/                           # Các LLM backbone khác
│   ├── llama-recipes/             # Llama2-7B adaptation
│   └── deepseek-r1/               # DeepSeek-R1 adaptation
│
├── agent/                         # Agent system (Qwen-Agent)
│   ├── assistant_qwen3.py         # Single-agent with Qwen3
│   ├── group_chat.py              # Multi-agent group chat
│   └── trafficllm_flask.py        # Flask API server
│
├── tutorials/                     # Hướng dẫn tạo PCAP
│   ├── generation.py              # Tạo packet với Scapy
│   └── config.json                # Config cho generation
│
├── images/                        # Hình ảnh, video demo
└── docs/                          # Tài liệu kỹ thuật
```

---

## 5. Chi tiết từng module

### 5.1 File gốc (root)

#### `config.json` — Bản đồ cấu hình trung tâm

```json
{
    "model_path": "D:\\models\\chatglm2-6b\\",     // Path đến model gốc
    "peft_path": "models/chatglm2/peft/",           // Thư mục chứa các adapter
    "peft_set": {                                    // Mapping: tên task → adapter path
        "NLP": "instruction/checkpoint-8000/",       // Stage 1 adapter
        "MTD": "ustc-tfc-2016-detection-.../",       // Malware Traffic Detection
        "BND": "iscx-botnet-2014-detection-.../",    // Botnet Detection
        ...
    },
    "tasks": {                                       // Mapping: tên task text → abbreviation
        "Malware Traffic Detection": "MTD",
        "Botnet Detection": "BND",
        ...
    }
}
```

**Luồng hoạt động:**
1. Stage 1 output `"Malware Traffic Detection"`
2. Tra `tasks` → `"MTD"`
3. Tra `peft_set["MTD"]` → `"ustc-tfc-2016-detection-packet/checkpoint-10000/"`
4. Load adapter từ path đó

#### `inference.py` — Inference dòng lệnh

```
python inference.py --config=config.json \
    --prompt="Please detect malware.<packet>frame.encap_type: 1, frame.time: ..."
```

Nhận input format: `instruction text + <packet> + traffic data`

#### `evaluation.py` — Đánh giá model

Đọc test dataset, chạy inference trên từng sample, so sánh output vs ground truth, tính metrics (accuracy, precision, recall, F1, confusion matrix).

#### `trafficllm_server.py` — Web demo (upload PCAP)

Streamlit app cho phép:
- Upload file PCAP
- Dùng `tshark` để parse PCAP → text features
- Nhập instruction
- Chạy dual-stage inference
- Hiển thị kết quả

#### `trafficllm_server_text.py` — Web demo (nhập text)

Giống server.py nhưng nhập traffic data dưới dạng text trực tiếp (không cần upload PCAP).

### 5.2 `preprocess/` — Tiền xử lý dữ liệu

**Mục đích:** Chuyển raw PCAP files → JSON dataset để train model.

**Luồng xử lý:**

```
Raw PCAP files (theo thư mục label)
    │
    ▼  preprocess_dataset.py
Chọn loại task (detection/generation/understanding)
    │
    ▼  packet_data_preprocess.py (hoặc flow_data_preprocess.py)
Dùng tshark/scapy để trích xuất features:
  - frame fields (time, length, protocols)
  - IP fields (src, dst, ttl, proto)
  - TCP/UDP fields (ports, flags, payload)
    │
    ▼  preprocess_utils.py
Ghép thành text format + thêm preprompt
    │
    ▼
Output: JSON file (1 dòng = 1 sample)
  {"instruction": "preprompt + <packet>: field1: val1, field2: val2, ...", "output": "label"}
```

**Packet features được trích xuất** (70+ fields):
- Frame: `encap_type`, `time`, `len`, `protocols`
- Ethernet: `src`, `dst`, `type`
- IP: `version`, `ttl`, `proto`, `src`, `dst`, `checksum`
- TCP: `srcport`, `dstport`, `flags`, `window_size`, `payload`
- UDP: `srcport`, `dstport`, `length`

**5 chế độ trích xuất packet:**

| Mode | Giải thích | Dùng cho |
|------|-----------|---------|
| `traffic words` | Dùng tshark trích xuất tất cả fields dạng text | Detection (mặc định) |
| `packet bytes` | Hex dump toàn bộ packet | Detection alternative |
| `packet words` | Scapy `.show()` output | Understanding |
| `generation 5tuple` | Chỉ lấy 5-tuple (src, dst, proto, sport, dport) | Generation headers |
| `generation data` | 5-tuple + hex payload | Generation full |

### 5.3 `tokenization/` — Tokenizer chuyên biệt

Dùng **SentencePiece BPE** train tokenizer mới trên traffic data:
- Vocab size: 64,794 tokens (match ChatGLM2)
- Training data: lấy từ instruction + output của datasets đã xử lý
- So sánh: TrafficLLM tokenizer tạo ra ít tokens hơn ChatGLM2 tokenizer cho cùng traffic data

### 5.4 `dual-stage-tuning/` — Training pipeline

**Stage 1 — Instruction Tuning** (`trafficllm_stage1.sh`):
- Data: `datasets/instructions/instruction.json`
- Mục đích: Dạy model hiểu instruction tiếng Anh → output tên task
- Hyperparameters: `pre_seq_len=128`, `lr=2e-2`, `max_steps=20000`

**Stage 2 — Traffic Tuning** (`trafficllm_stage2.sh`):
- Data: Dataset cụ thể cho từng task (ví dụ: `ustc-tfc-2016_detection_packet_train.json`)
- Mục đích: Dạy model nhận traffic data → output label
- Hyperparameters: tương tự Stage 1

**Training script** (`main.py`):
- Dựa trên HuggingFace `Seq2SeqTrainer`
- Hỗ trợ P-Tuning v2 (prefix encoder)
- Metrics: ROUGE, BLEU-4 (cho generation), hoặc custom cho detection
- Distributed training qua `torchrun`

### 5.5 `EA-PEFT/` — Mở rộng và cập nhật model

EA-PEFT cho phép:
- **Update**: Cập nhật model cho task đã có (khi traffic pattern thay đổi)
- **Register**: Thêm task hoàn toàn mới

```python
# Update model cho Malware Traffic Detection
python ea-peft.py --model_name /path/model --tuning_data /path/data --adaptation_task update --task_name MTD

# Đăng ký task mới
python ea-peft.py --model_name /path/model --tuning_data /path/data --adaptation_task register --task_name NEW_TASK
```

### 5.6 `datasets/` — Dữ liệu huấn luyện

Mỗi dataset có 3 files:
- `*_train.json`: Dữ liệu training (JSONL format)
- `*_test.json`: Dữ liệu testing
- `*_label.json`: Mapping label name → label ID

**Ví dụ 1 sample trong `ustc-tfc-2016_detection_packet_test.json`:**
```json
{
  "instruction": "Given the following traffic data <packet> that contains protocol fields...
    <packet>: frame.encap_type: 1, frame.time: Aug 10, 2011, ip.src: 147.32.84.165,
    tcp.dstport: 80, tcp.payload: 47:45:54:20:2f:70:36...",
  "output": "Neris"
}
```

**Label file (`ustc-tfc-2016_label.json`):**
```json
{
    "Geodo": 0, "Cridex": 1, "Tinba": 2, ...,
    "BitTorrent": 11, "Skype": 12, ..., "Neris": 19
}
```

### 5.7 `models/` — Checkpoints đã train

```
models/chatglm2/peft/
├── instruction/checkpoint-8000/           # NLP adapter (Stage 1)
│   ├── pytorch_model.bin                  # Chỉ ~2MB (prefix encoder weights)
│   ├── config.json, tokenizer.model...
│
├── ustc-tfc-2016-detection-packet/        # MTD adapter (Stage 2)
│   └── checkpoint-10000/
├── iscx-botnet-2014-detection-packet/     # BND adapter
│   └── checkpoint-5000/
├── csic-2010-detection-packet/            # WAD adapter
│   └── checkpoint-6000/
└── ...
```

Mỗi checkpoint chứa:
- `pytorch_model.bin`: Trọng số adapter (nhỏ, ~2-10MB)
- `config.json`: Cấu hình model
- `trainer_state.json`: Trạng thái training (loss history, steps...)
- `tokenizer.model`: Tokenizer

### 5.8 `tutorials/` — Tạo traffic với Scapy

`generation.py` cho phép **sinh packet giả** từ TrafficLLM:

```
Input: "Please generate a packet of Neris traffic."

Stage 1: → "Malware Traffic Generation" (task = MTG)
Stage 2a: Header adapter → {'src': '147.32.84.165', 'dst': '212.117.171.138', 'proto': 6, ...}
Stage 2b: Payload adapter → "2080008070002000706001..."

→ Scapy tạo Ether/IP/TCP packet
→ text2pcap tạo file synthetic_packet.pcap
→ Mở được bằng Wireshark!
```

### 5.9 `Adapt2BitNet/` — Backbone BitNet 1.58-bit

**BitNet b1.58-2B-4T** là model 2B tham số với trọng số ternary ({-1, 0, +1}):
- Nhỏ hơn 3x so với ChatGLM2 (2B vs 6B)
- Sử dụng LoRA thay vì P-Tuning v2
- `AutoModelForCausalLM` + `model.generate()` thay vì `.chat()`
- Cần `transformers>=4.57` và `peft>=0.7.0`

So sánh với ChatGLM2 path:

| Aspect | ChatGLM2 | BitNet |
|--------|----------|--------|
| Model size | 6B | 2B |
| PEFT method | P-Tuning v2 | LoRA |
| Inference API | `.chat()` | `model.generate()` |
| Adapter loading | Manual state_dict swap | `PeftModel.from_pretrained()` |
| VRAM inference | ~16GB | ~8GB |

### 5.10 `Adapt2GLM4/` — Backbone GLM4

GLM4-9B-Chat: phiên bản mới hơn của ChatGLM, nhanh hơn, chính xác hơn.
- Sử dụng LoRA
- Cần `transformers==4.44.0`

### 5.11 `llm/` — Backbone Llama2 & DeepSeek-R1

- **Llama2-7B**: Dùng classification head + LoRA, cross-entropy loss
- **DeepSeek-R1**: Chain-of-thought reasoning cho traffic analysis

### 5.12 `agent/` — Hệ thống Agent

Dựa trên [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent), cho phép xây dựng:

**Single-Agent:**
```
Qwen3 (tool-calling LLM)
    ├── trafficllm tool (gọi TrafficLLM Flask API)
    ├── security_qa tool (gọi SecGPT cho Q&A bảo mật)
    ├── code_interpreter (chạy Python code)
    ├── MCP: time server
    └── MCP: fetch server
```

**Multi-Agent (Group Chat):**
Nhiều agent cộng tác, mỗi agent chuyên một nhiệm vụ.

---

## 6. Luồng dữ liệu end-to-end

```
                           TRAINING PHASE
                           ==============

Raw PCAP files                    Human-written instructions
(labeled folders)                 (9,209 samples)
        │                                 │
        ▼                                 ▼
   preprocess/                    datasets/instructions/
   preprocess_dataset.py          instruction.json
        │                                 │
        ▼                                 │
   datasets/                              │
   *_train.json, *_test.json              │
        │                                 │
        ▼                                 ▼
   ┌────────────┐               ┌────────────────┐
   │  Stage 2   │               │    Stage 1     │
   │  Traffic   │               │  Instruction   │
   │  Tuning    │               │    Tuning      │
   └─────┬──────┘               └───────┬────────┘
         │                              │
         ▼                              ▼
   models/peft/                  models/peft/
   task-specific/                instruction/
   checkpoint-N/                 checkpoint-N/


                          INFERENCE PHASE
                          ===============

User sends: instruction + PCAP file (or text)
        │
        ▼ (tshark parses PCAP → text features)
        │
        ▼
   ┌──────────────────────┐
   │ Load base model      │  (ChatGLM2-6B / BitNet / GLM4)
   │ from config.json     │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ Stage 1: NLP adapter │
   │ Input: instruction   │
   │ Output: task name    │  → "Malware Traffic Detection"
   └──────────┬───────────┘
              │ lookup config.json → "MTD"
              ▼
   ┌──────────────────────┐
   │ Stage 2: MTD adapter │
   │ Input: preprompt +   │
   │   traffic features   │
   │ Output: prediction   │  → "Zeus"
   └──────────┬───────────┘
              │
              ▼
   Display result to user
```

---

## 7. Các downstream task cụ thể

### Traffic Detection Tasks (10 tasks)

| Abbrev. | Task | Input | Output | Dataset |
|---------|------|-------|--------|---------|
| **MTD** | Malware Traffic Detection | Encrypted packet data | Malware family (Zeus, Neris...) hoặc benign app (Gmail, Skype...) | USTC TFC 2016 (50.7K) |
| **BND** | Botnet Detection | Packet data | Botnet type (IRC, Neris, RBot, Virut) hoặc "normal" | ISCX Botnet 2014 (25K) |
| **MDD** | Malicious DoH Detection | DoH packet data | "malicious" hoặc "benign" | DoHBrw 2020 (47.8K) |
| **WAD** | Web Attack Detection | HTTP request (method, URL, body) dạng JSON | "malicious" hoặc "benign" | CSIC 2010 (34.5K) |
| **AAD** | APT Attack Detection | HTTP request dạng JSON | "abnormal" hoặc "normal" | DAPT 2020 (10K) |
| **EVD** | Encrypted VPN Detection | VPN encrypted packet | App category (facebook, youtube, skype...) | ISCX VPN 2016 (64.8K) |
| **TBD** | Tor Behavior Detection | Tor traffic packet | Behavior (browsing, chat, video...) | ISCX Tor 2016 (40K) |
| **EAC** | Encrypted App Classification | Encrypted packet | App name | CSTNET 2023 (97.6K) |
| **WF** | Website Fingerprinting | Traffic features | Website category | CW-100 2018 (7.4K) |
| **CD** | Concept Drift | Traffic packet | App label (xét cả thay đổi pattern) | APP-53 2023 (109.8K) |

### Traffic Generation Tasks (4 tasks)

| Abbrev. | Task | Input | Output |
|---------|------|-------|--------|
| **MTG** | Malware Traffic Generation | "Generate Neris traffic" | Header dict + hex payload |
| **BTG** | Botnet Traffic Generation | "Generate botnet traffic" | Header dict + hex payload |
| **EVG** | Encrypted VPN Generation | "Generate VPN traffic" | Header dict + hex payload |
| **EAG** | Encrypted App Generation | "Generate app traffic" | Header dict + hex payload |

---

## 8. Datasets chi tiết

### Instruction Dataset

Gồm 9,209 câu instruction tiếng Anh, mỗi câu map đến 1 task name:

```json
{"instruction": "I have a copy of traffic that may be botnet traffic...", "output": "Botnet Detection"}
{"instruction": "What type of application might this traffic belong to?", "output": "Malware Traffic Detection"}
{"instruction": "Based on encrypted app traffic, create a packet of Arxiv type.", "output": "Encrypted App Generation"}
```

### Traffic Datasets

Mỗi dataset gồm:
- **`*_train.json`**: JSONL, mỗi dòng = 1 JSON object
- **`*_test.json`**: Tương tự, dùng để test
- **`*_label.json`**: Map tên label → số ID

**Format mỗi sample:**
```json
{
  "instruction": "[preprompt]\n<packet>: [extracted features]",
  "output": "[label]"
}
```

**Preprompt** là đoạn text cố định mô tả task, ví dụ:
> "Given the following traffic data \<packet\> that contains protocol fields, traffic features, and payloads. Please conduct the ENCRYPTED MALWARE DETECTION TASK to determine which application category..."

---

## 9. Config.json — Bản đồ cấu hình

### ChatGLM2 config (root `config.json`)

```json
{
    "model_path": "path/to/chatglm2-6b/",        // Base model
    "peft_path": "models/chatglm2/peft/",          // Adapter root
    "peft_set": {
        "NLP": "instruction/checkpoint-8000/",      // Stage 1
        "MTD": "ustc-tfc-2016-.../checkpoint-10000/", // Stage 2 per task
        "BND": "iscx-botnet-.../checkpoint-5000/",
        "WAD": "csic-2010-.../checkpoint-6000/",
        "AAD": "dapt-2020-.../checkpoint-20000/",
        "EVD": "iscx-vpn-2016-.../checkpoint-4000/",
        "TBD": "iscx-tor-2016-.../checkpoint-10000/"
    },
    "tasks": {
        "Malware Traffic Detection": "MTD",         // Stage 1 output → abbreviation
        "Botnet Detection": "BND",
        "Web Attack Detection": "WAD",
        "APT Attack Detection": "AAD",
        "Encrypted VPN Detection": "EVD",
        "Tor Behavior Detection": "TBD"
    }
}
```

### BitNet config (`Adapt2BitNet/config.json`)

```json
{
    "model_path": "microsoft/bitnet-b1.58-2B-4T-bf16",  // HuggingFace model ID
    "adapter_path": "models/bitnet/adapters/",            // LoRA adapters root
    "adapter_set": { ... },                               // Same structure as peft_set
    "tasks": { ... }                                      // Same structure
}
```

---

## 10. Cách chạy từng phần

### 10.1 Cài đặt môi trường

```bash
conda create -n trafficllm python=3.9
conda activate trafficllm
pip install -r requirements.txt

# Nếu training
pip install rouge_chinese nltk jieba datasets
```

### 10.2 Download model

- ChatGLM2-6B: https://huggingface.co/THUDM/chatglm2-6b
- Fine-tuned adapters: https://drive.google.com/drive/folders/1YjEhdordqZRpnw_oKczwUztcT52T0oQ0
- Datasets: https://drive.google.com/drive/folders/1RZAOPcNKq73-quA8KG_lkAo_EqlwhlQb

### 10.3 Preprocessing raw traffic

```bash
cd preprocess
python preprocess_dataset.py \
    --input /path/to/raw/pcap/folders \
    --dataset_name ustc-tfc-2016 \
    --traffic_task detection \
    --granularity packet \
    --output_path /output/path \
    --output_name ustc-tfc-2016
```

### 10.4 Training

```bash
cd dual-stage-tuning

# Stage 1: Instruction tuning
bash trafficllm_stage1.sh

# Stage 2: Traffic tuning (per task)
bash trafficllm_stage2.sh
```

### 10.5 Evaluation

```bash
python evaluation.py \
    --model_name /path/to/chatglm2-6b \
    --traffic_task detection \
    --test_file datasets/ustc-tfc-2016/ustc-tfc-2016_detection_packet_test.json \
    --label_file datasets/ustc-tfc-2016/ustc-tfc-2016_label.json \
    --ptuning_path models/chatglm2/peft/ustc-tfc-2016-detection-packet/checkpoint-10000/
```

### 10.6 Inference (Terminal)

```bash
python inference.py --config=config.json \
    --prompt="Please detect malware.<packet>frame.encap_type: 1, frame.time: ..."
```

### 10.7 Web Demo

```bash
streamlit run trafficllm_server.py
# Truy cập http://localhost:8501
```

### 10.8 Traffic Generation

```bash
cd tutorials
mkdir -p tmp/packet_generation tmp/flow_generation
python generation.py --config=config.json \
    --prompt='Please generate a packet of Neris traffic.'
# Output: synthetic_packet.pcap
```

---

## 11. Yêu cầu phần cứng

| Thao tác | GPU VRAM | RAM | Lưu ý |
|----------|----------|-----|-------|
| **Inference ChatGLM2** | ~16GB | 16GB+ | Cần NVIDIA GPU với CUDA |
| **Training ChatGLM2** | ~24GB+ | 32GB+ | A100/A6000 khuyến nghị |
| **Inference BitNet** | ~8GB | 16GB+ | Nhẹ hơn đáng kể |
| **Training BitNet (LoRA)** | ~16GB | 32GB+ | BF16 precision |
| **Preprocessing** | Không cần GPU | 8GB+ | Cần cài tshark |
| **Web Demo** | Giống inference | 16GB+ | Cần cài Streamlit |

---

## 12. Thuật ngữ (Glossary)

| Thuật ngữ | Giải thích |
|-----------|-----------|
| **LLM** | Large Language Model — mô hình ngôn ngữ lớn |
| **Token** | Đơn vị nhỏ nhất mà model xử lý (1 từ hoặc phần từ) |
| **Fine-tuning** | Huấn luyện thêm model đã train trên data mới |
| **PEFT** | Parameter-Efficient Fine-Tuning — fine-tune tiết kiệm |
| **P-Tuning v2** | Kỹ thuật PEFT thêm prefix trainable vào model |
| **LoRA** | Low-Rank Adaptation — kỹ thuật PEFT thêm ma trận nhỏ |
| **Adapter** | File trọng số nhỏ chứa "kỹ năng" đã train cho 1 task |
| **Checkpoint** | Snapshot trọng số model tại 1 thời điểm training |
| **Inference** | Chạy model để dự đoán trên data mới |
| **Preprompt** | Đoạn text mô tả task, ghép trước traffic data |
| **Dual-Stage** | 2 giai đoạn: hiểu instruction → phân tích traffic |
| **PCAP** | Packet Capture — file ghi lại network traffic |
| **Packet** | Một gói tin đơn lẻ trên mạng |
| **Flow** | Chuỗi packets thuộc cùng 1 kết nối |
| **5-tuple** | (src IP, dst IP, src port, dst port, protocol) — định danh flow |
| **Payload** | Nội dung data thực sự trong packet (sau header) |
| **tshark** | Command-line tool phân tích PCAP (phần của Wireshark) |
| **Scapy** | Thư viện Python tạo/phân tích network packets |
| **BPE** | Byte Pair Encoding — thuật toán tokenization |
| **CausalLM** | Causal Language Model — model sinh text từ trái qua phải |
| **Seq2Seq** | Sequence-to-Sequence — model nhận chuỗi input, sinh chuỗi output |
| **MCP** | Model Context Protocol — giao thức kết nối tool cho agent |
| **EA-PEFT** | Extensible Adaptation PEFT — mở rộng model cho task mới |
| **Confusion Matrix** | Ma trận hiển thị kết quả phân loại chi tiết |
| **BF16** | Brain Float 16 — format số 16-bit cho training |
| **Ternary Weights** | Trọng số chỉ có 3 giá trị: {-1, 0, +1} (BitNet) |
| **GQA** | Grouped Query Attention — kỹ thuật attention hiệu quả |

---

## 13. Phụ lục: Các khái niệm thuật toán & kỹ thuật Deep Learning

> Phần này giải thích chi tiết **tất cả thuật toán và kỹ thuật** được sử dụng trong repo TrafficLLM, từ nền tảng đến nâng cao.

### 13.1 Transformer Architecture

Transformer là **kiến trúc nền tảng** của mọi LLM trong repo (ChatGLM2, BitNet, Llama, GLM4). Được giới thiệu trong paper "Attention Is All You Need" (2017).

**Cấu trúc cơ bản:**

```
Input Tokens  [x₁, x₂, x₃, ...]
     │
     ▼
┌──────────────────────┐
│   Token Embedding    │  Chuyển token ID → vector số thực (ví dụ: 4096 chiều)
│   + Positional Enc.  │  Thêm thông tin vị trí
└──────────┬───────────┘
           │
     ┌─────┴─────┐ ← Lặp lại N lần (N=28 cho ChatGLM2, N=30 cho BitNet)
     │           │
     ▼           │
┌──────────────┐ │
│ Self-Attention│ │  Mỗi token "nhìn" tất cả token khác để hiểu ngữ cảnh
│   Layer      │ │
└──────┬───────┘ │
       │         │
       ▼         │
┌──────────────┐ │
│  Feed-Forward│ │  Mạng neural 2 lớp xử lý từng vị trí
│   Network    │ │
└──────┬───────┘ │
       │         │
       └─────────┘
           │
           ▼
┌──────────────────────┐
│   Output Layer       │  Chuyển vector → xác suất cho mỗi token tiếp theo
│   (Vocabulary Head)  │  (128K token cho BitNet, 64K cho ChatGLM2)
└──────────────────────┘
```

**Tại sao Transformer mạnh?** Nhờ cơ chế **Attention** cho phép model xem xét mối quan hệ giữa **mọi cặp token** trong input, bất kể khoảng cách.

### 13.2 Self-Attention Mechanism

Self-Attention là **trái tim** của Transformer. Nó cho phép mỗi token "hỏi" các token khác: "Bạn có liên quan đến tôi không?"

**Thuật toán (Scaled Dot-Product Attention):**

```
Mỗi token tạo ra 3 vector từ embedding của nó:
  Q (Query)  = "Tôi đang tìm gì?"
  K (Key)    = "Tôi có thông tin gì?"
  V (Value)  = "Thông tin thực tế của tôi"

Bước 1: Tính điểm attention
  Score = Q × Kᵀ / √d_k

  Ví dụ với 4 tokens:
  Score = ┌                    ┐
          │ q₁·k₁  q₁·k₂  q₁·k₃  q₁·k₄ │  ← token 1 "hỏi" tất cả
          │ q₂·k₁  q₂·k₂  q₂·k₃  q₂·k₄ │  ← token 2 "hỏi" tất cả
          │ q₃·k₁  q₃·k₂  q₃·k₃  q₃·k₄ │
          │ q₄·k₁  q₄·k₂  q₄·k₃  q₄·k₄ │
          └                    ┘

Bước 2: Softmax → trọng số attention (tổng mỗi hàng = 1)
  Weights = softmax(Score)

Bước 3: Tổng hợp
  Output = Weights × V
```

**Ví dụ trực giác** với câu "The cat sat on the mat":
- Khi xử lý "sat", attention cao nhất vào "cat" (chủ ngữ) → model hiểu "cat" là thứ thực hiện hành động "sat"

**Trong traffic data:** Khi xử lý `tcp.dstport: 443`, attention có thể cao vào `frame.protocols: eth:ip:tcp:ssl` → model hiểu đây là kết nối TLS/SSL.

### 13.3 Multi-Head Attention & Grouped Query Attention

**Multi-Head Attention (MHA):**

Thay vì chạy attention 1 lần, chia Q, K, V thành nhiều "đầu" (heads) chạy song song:

```
Embedding (4096 chiều)
    │
    ├── Head 1 (128 chiều): chú ý đến cú pháp
    ├── Head 2 (128 chiều): chú ý đến ngữ nghĩa
    ├── Head 3 (128 chiều): chú ý đến vị trí
    ├── ... (32 heads tổng cộng)
    └── Head 32 (128 chiều)
    │
    ▼ Nối lại
Output (4096 chiều)
```

Mỗi head học được một "góc nhìn" khác nhau.

**Grouped Query Attention (GQA)** — dùng trong BitNet:

Thay vì mỗi head có Q, K, V riêng, GQA **chia sẻ K và V** giữa nhiều head:

```
MHA chuẩn (32 heads):     32 Q-heads, 32 K-heads, 32 V-heads
GQA (BitNet: 20Q, 5KV):   20 Q-heads, 5 K-heads, 5 V-heads
                           → mỗi 4 Q-heads chia sẻ 1 K-head và 1 V-head
```

Lợi ích: **giảm bộ nhớ** (ít K,V cần lưu) và **tăng tốc inference** mà gần như không mất chất lượng.

### 13.4 Positional Encoding / Embedding

Transformer xử lý tất cả tokens **đồng thời** (parallel), nên nó không tự biết thứ tự. Positional encoding thêm thông tin "token này ở vị trí thứ mấy."

```
Token embedding:       [0.5, -0.3, 0.8, ...]    ← ý nghĩa của từ
Positional encoding:   [0.0, 1.0, 0.0, ...]     ← vị trí trong câu
                       ─────────────────────
Tổng:                  [0.5, 0.7, 0.8, ...]      ← input cho Transformer
```

Các biến thể:
- **Sinusoidal** (Transformer gốc): dùng hàm sin/cos
- **Learned** (ChatGLM2): học position embedding trong training
- **RoPE** (Rotary Position Embedding, dùng trong BitNet/Llama): xoay vector Q, K theo vị trí

### 13.5 Feed-Forward Network (FFN) & Gate Mechanism

Sau self-attention, mỗi token đi qua một **mạng neural 2 lớp**:

```
FFN chuẩn:
  output = W₂ × ReLU(W₁ × input + b₁) + b₂

  W₁: [4096 → 16384]   ← mở rộng 4x
  W₂: [16384 → 4096]   ← thu nhỏ lại
```

**SwiGLU FFN** (dùng trong BitNet, Llama):

```
SwiGLU:
  gate   = Swish(W_gate × input)     ← "cổng" kiểm soát thông tin
  up     = W_up × input               ← thông tin thực
  output = W_down × (gate ⊙ up)       ← ⊙ = nhân từng phần tử

  → 3 ma trận: gate_proj, up_proj, down_proj
```

Đây chính là lý do trong LoRA config của BitNet có `target_modules: ["gate_proj", "up_proj", "down_proj"]`.

### 13.6 Causal LM vs Seq2Seq (Encoder-Decoder)

Repo sử dụng **2 paradigm khác nhau** tùy backbone:

**Causal LM (Decoder-only)** — BitNet, Llama, GLM4:

```
Input:  [token₁] [token₂] [token₃]  →  [token₄]  →  [token₅]
         ↓        ↓        ↓             ↓
        Mỗi token chỉ "nhìn" được các token TRƯỚC nó
        (causal mask: token₃ thấy token₁,₂,₃ nhưng KHÔNG thấy token₄,₅)
```

Khi sinh text:
```
"Given traffic data..."  →  model sinh →  "Z"  →  "e"  →  "u"  →  "s"  →  [EOS]
                             (autoregressive: từng token một)
```

**Seq2Seq (Encoder-Decoder)** — ChatGLM2:

```
Encoder: đọc toàn bộ input cùng lúc (bidirectional)
   "Given traffic data <packet>: frame.time: ..."
              ↓
Decoder: sinh output từng token (causal)
   "Zeus" (token by token)
```

ChatGLM2 dùng biến thể gọi là **Prefix LM**: phần prefix (input) được xử lý bidirectional, phần sinh (output) là causal.

### 13.7 Loss Function: Cross-Entropy

Trong training, model cần biết nó **sai bao nhiêu** để tự sửa. Cross-entropy loss đo khoảng cách giữa phân phối dự đoán và đáp án đúng.

```
Ví dụ: Model cần dự đoán token tiếp theo là "Zeus" (ID=8)

Model output (xác suất qua softmax):
  P("Zeus")     = 0.6   ← model khá tự tin
  P("Neris")    = 0.2
  P("Cridex")   = 0.1
  P(các từ khác) = 0.1

Loss = -log(P("Zeus")) = -log(0.6) = 0.51

Nếu P("Zeus") = 0.99 → Loss = 0.01  (rất tốt!)
Nếu P("Zeus") = 0.01 → Loss = 4.61  (rất tệ!)
```

Trong code training (`main.py`), label padding token (`-100`) được bỏ qua khi tính loss → model không bị phạt cho phần input, chỉ bị phạt cho phần output sai.

### 13.8 Backpropagation & Gradient Descent

**Backpropagation** là thuật toán tính "model nên thay đổi mỗi tham số bao nhiêu và theo hướng nào" để giảm loss.

```
Forward pass (tính loss):
  Input → Model → Output → So sánh với đáp án → Loss = 0.51

Backward pass (tính gradient):
  Loss → đạo hàm ngược qua từng lớp → gradient cho mỗi tham số
  ∂Loss/∂W cho mỗi weight W

Update (gradient descent):
  W_mới = W_cũ - learning_rate × gradient
```

**Trực giác:** Bạn đứng trên đỉnh núi, cần đi xuống thung lũng (loss thấp nhất):
- Gradient = hướng dốc nhất
- Learning rate = kích thước bước chân
- Mỗi bước = 1 training step

### 13.9 Learning Rate, Warmup & Scheduler

**Learning Rate (LR):** Tốc độ học — bước chân khi đi xuống núi.

```
LR quá lớn (0.1):  ╱╲╱╲╱╲  → nhảy qua nhảy lại, không hội tụ
LR vừa (2e-2):     ╲  ╱     → giảm dần, tìm được điểm tốt
LR quá nhỏ (1e-8): ─────    → gần như đứng yên, train mãi không xong
```

**Warmup** (dùng trong BitNet config: `warmup_steps: 200`):

```
LR   ↑     /‾‾‾‾‾‾‾‾‾\___________
     │    /                        \_____
     │   /
     │  /
     └──┴────────────────────────────────→ Steps
     0  200                            10000

Giai đoạn 0-200: LR tăng dần từ 0 (warmup)
Giai đoạn 200+: LR giảm dần (decay)
```

Warmup giúp model ổn định ở đầu training, tránh gradient quá lớn phá hỏng trọng số.

**Trong code:**
- ChatGLM2: `learning_rate = 2e-2` (P-Tuning v2 dùng LR lớn vì chỉ train prefix nhỏ)
- BitNet LoRA: `learning_rate = 5e-5` (LoRA dùng LR nhỏ hơn vì ảnh hưởng trực tiếp đến output)

### 13.10 Gradient Accumulation

**Vấn đề:** GPU nhỏ không đủ VRAM để train batch_size lớn (ví dụ batch=16 cần 48GB).

**Giải pháp:** Tích lũy gradient qua nhiều mini-batch nhỏ trước khi cập nhật:

```
gradient_accumulation_steps = 16, per_device_batch_size = 1

Step 1: xử lý 1 sample, TÍCH LŨY gradient (chưa update)
Step 2: xử lý 1 sample, TÍCH LŨY gradient (chưa update)
...
Step 16: xử lý 1 sample, TÍCH LŨY gradient
→ CẬP NHẬT trọng số (equivalent batch size = 1 × 16 = 16)
```

Trong `dual-stage-tuning/trafficllm_stage1.sh`:
```
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16
```
→ Effective batch size = 16, nhưng chỉ cần VRAM cho batch=1.

### 13.11 Gradient Checkpointing

**Vấn đề:** Backpropagation cần lưu tất cả activations (giá trị trung gian) từ forward pass → tốn rất nhiều VRAM.

**Giải pháp:** Chỉ lưu activations tại một số "checkpoint" layers. Khi cần gradient cho layer đã xóa → **tính lại** từ checkpoint gần nhất.

```
Bình thường: Lưu activation MỌI layer → VRAM lớn
  Layer 1 → [lưu] → Layer 2 → [lưu] → ... → Layer 28 → [lưu]

Gradient Checkpointing: Chỉ lưu một vài
  Layer 1 → [lưu] → Layer 2 → [xóa] → ... → Layer 7 → [lưu] → ...
  Khi cần Layer 2: tính lại từ Layer 1 (chậm hơn, nhưng tiết kiệm VRAM)
```

Trong code (`main.py`, `finetune.py`):
```python
model.gradient_checkpointing_enable()
```

**Trade-off:** Giảm ~60% VRAM, tăng ~20% thời gian training.

### 13.12 Mixed Precision Training (FP16 / BF16)

Mỗi số thực trong model được lưu bằng một format nhất định:

```
FP32 (32-bit): ±1.xxxxxxxxxxxxxxxxxxxx × 2^±yyy   (chính xác cao, tốn VRAM)
FP16 (16-bit): ±1.xxxxxxxxxx × 2^±yyyyyy          (nhanh, nhưng dễ overflow)
BF16 (16-bit): ±1.xxxxxxx × 2^±yyyyyyyy           (range rộng hơn FP16, ổn định hơn)
```

| Format | Bits | Range | Dùng ở đâu |
|--------|------|-------|-------------|
| FP32 | 32 | ±3.4×10³⁸ | Gradient accumulation, optimizer states |
| FP16 | 16 | ±65504 | Forward/backward pass (ChatGLM2: `model.half()`) |
| BF16 | 16 | ±3.4×10³⁸ | BitNet training (`bf16: true` trong lora.yaml) |

**Mixed precision** = dùng FP16/BF16 cho tính toán nhanh, giữ FP32 cho phần quan trọng (master weights, loss scaling).

Trong code:
- ChatGLM2: `model = model.half()` → FP16, nhưng `prefix_encoder.float()` → prefix giữ FP32
- BitNet: `torch_dtype=torch.bfloat16` → BF16

### 13.13 BPE (Byte Pair Encoding) Tokenization

BPE là thuật toán **xây dựng từ điển** (vocabulary) cho tokenizer.

**Thuật toán:**

```
Bước 0: Bắt đầu với từng ký tự
  Corpus: "tcp tcp udp"
  Vocab: {t, c, p, u, d, _}
  Tokens: [t, c, p, _, t, c, p, _, u, d, p]

Bước 1: Tìm cặp xuất hiện nhiều nhất → "t,c" (3 lần) → merge thành "tc"
  Vocab: {t, c, p, u, d, _, tc}
  Tokens: [tc, p, _, tc, p, _, u, d, p]

Bước 2: Tìm cặp nhiều nhất → "tc,p" (2 lần) → merge thành "tcp"
  Vocab: {t, c, p, u, d, _, tc, tcp}
  Tokens: [tcp, _, tcp, _, u, d, p]

Bước 3: "u,d" → "ud"
Bước 4: "ud,p" → "udp"
  Vocab: {t, c, p, u, d, _, tc, tcp, ud, udp}
  Tokens: [tcp, _, tcp, _, udp]
```

Lặp lại cho đến khi vocab đạt kích thước mong muốn (64,794 trong TrafficLLM).

**Tại sao BPE cho traffic?** Dữ liệu traffic có nhiều pattern lặp lại (hex payload, field names) mà tokenizer gốc không biết. BPE train trên traffic data sẽ tạo ra tokens chuyên biệt → ít tokens hơn = context window chứa nhiều thông tin hơn.

Trong `tokenization/traffic_tokenizer.py`:
```python
spm.SentencePieceTrainer.Train(
    input="dataset.txt",
    vocab_size=64794,
    model_type="bpe",     # ← Byte Pair Encoding
)
```

### 13.14 Prefix Tuning / P-Tuning v2 (thuật toán chi tiết)

Đây là PEFT method dùng cho **ChatGLM2 path**.

**Ý tưởng:** Thêm K "virtual tokens" (prefix) vào đầu mỗi layer attention. Chỉ train các virtual tokens này.

```
Bình thường (input có 5 tokens):
  Attention input: [x₁, x₂, x₃, x₄, x₅]

Với P-Tuning v2 (pre_seq_len=128):
  Attention input: [p₁, p₂, ..., p₁₂₈, x₁, x₂, x₃, x₄, x₅]
                    ├── 128 virtual ──┤ ├── 5 real tokens ──┤
                    (TRAINABLE)         (FROZEN)
```

**Chi tiết kỹ thuật:**

```
Prefix Encoder (trainable):
  embedding = nn.Embedding(pre_seq_len, hidden_size)
  → Shape: [128, 4096] cho ChatGLM2
  → Tổng params: 128 × 4096 = 524K (rất nhỏ so với model 6B)

Khi inference, prefix tokens tham gia attention:
  Q = [q_p₁...q_p₁₂₈, q_x₁...q_x₅]   (query từ prefix + real)
  K = [k_p₁...k_p₁₂₈, k_x₁...k_x₅]   (key từ prefix + real)
  V = [v_p₁...v_p₁₂₈, v_x₁...v_x₅]   (value từ prefix + real)

  → Real tokens x₁...x₅ "nhìn" thấy cả prefix → bị prefix "điều khiển"
```

Trong code (`inference.py`, `evaluation.py`):
```python
config = AutoConfig.from_pretrained(model_path, pre_seq_len=128)  # ← 128 prefix tokens
model = AutoModel.from_pretrained(model_path, config=config)

# Load trained prefix weights
prefix_state_dict = torch.load("pytorch_model.bin")
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
```

### 13.15 LoRA — Low-Rank Adaptation (thuật toán chi tiết)

Dùng trong **BitNet** và **GLM4 path**.

**Toán học:**

Mỗi lớp linear gốc: `y = Wx` với W có shape `[d_out, d_in]` (ví dụ: [4096, 4096])

LoRA thêm:
```
y = Wx + (B × A)x

Trong đó:
  W: [4096, 4096]  → FROZEN (16M params, không train)
  A: [r, 4096]     → TRAINABLE (r=8 → 32K params)
  B: [4096, r]     → TRAINABLE (r=8 → 32K params)

Tổng trainable: 64K params (thay vì 16M)
Tỉ lệ: 0.4% parameters!
```

**Tại sao hoạt động?** Nghiên cứu cho thấy sự thay đổi trọng số khi fine-tune có **rank thấp** (low-rank) — tức là có thể biểu diễn bằng tích 2 ma trận nhỏ mà gần như không mất thông tin.

**Scaling:**
```
y = Wx + (α/r) × BAx

α (lora_alpha) = 32, r = 8
→ scale factor = 32/8 = 4
→ LoRA update được khuếch đại 4x
```

Trong `Adapt2BitNet/FT/configs/lora.yaml`:
```yaml
peft_config:
  peft_type: LORA
  task_type: CAUSAL_LM
  r: 8                    # rank
  lora_alpha: 32           # scaling factor
  lora_dropout: 0.1        # dropout để regularization
  target_modules:          # áp dụng LoRA lên lớp nào
    - q_proj               # Query projection (Attention)
    - k_proj               # Key projection (Attention)
    - v_proj               # Value projection (Attention)
    - o_proj               # Output projection (Attention)
    - gate_proj            # Gate FFN (SwiGLU)
    - up_proj              # Up FFN (SwiGLU)
    - down_proj            # Down FFN (SwiGLU)
```

**Adapter hot-swap** (trong `inference_bitnet.py`):
```python
# Load NLP adapter
model_nlp = PeftModel.from_pretrained(base_model, nlp_adapter_path)

# Unload để giải phóng
model_nlp = model_nlp.unload()

# Load task adapter
model_task = PeftModel.from_pretrained(base_model, task_adapter_path)
```

### 13.16 Text Generation: Temperature, Top-p, Top-k, Beam Search

Khi model sinh text, tại mỗi bước nó dự đoán **phân phối xác suất** trên toàn bộ vocabulary. Các tham số sau kiểm soát cách chọn token tiếp theo:

**Temperature (nhiệt độ):**

```
Logits gốc:    [2.0, 1.0, 0.5, 0.1, ...]

Temperature=1.0 (mặc định):
  Softmax → [0.45, 0.17, 0.10, 0.07, ...]    (phân phối vừa phải)

Temperature=0.1 (gần deterministic):
  Logits/0.1 = [20.0, 10.0, 5.0, 1.0, ...]
  Softmax → [0.99, 0.005, 0.001, ...]         (hầu như chỉ chọn top 1)

Temperature=2.0 (sáng tạo hơn):
  Logits/2.0 = [1.0, 0.5, 0.25, 0.05, ...]
  Softmax → [0.30, 0.18, 0.14, 0.12, ...]     (phân bố đều hơn, random hơn)
```

- **Detection tasks:** dùng `temperature=0.1` → output ổn định, deterministic
- **Generation tasks:** dùng `temperature=0.7-0.8` → đa dạng hơn

**Top-p (Nucleus Sampling):**

Chỉ sample từ **tập nhỏ nhất** các tokens mà tổng xác suất ≥ p:

```
Xác suất:  Token_A=0.5  Token_B=0.2  Token_C=0.15  Token_D=0.1  Token_E=0.05

top_p=0.85:
  Tổng tích lũy: A(0.5) + B(0.7) + C(0.85) ✓
  → Chỉ sample từ {A, B, C}, bỏ qua D, E (ít khả năng)
```

**Beam Search:**

Thay vì chọn 1 token tốt nhất tại mỗi bước (greedy), beam search giữ **K candidates** (beams) tốt nhất:

```
Beam width = 3:

Step 1: "The"  →  cat(0.3), dog(0.25), bird(0.2)
Step 2: "The cat"    → sat(0.4), ran(0.3)
        "The dog"    → barked(0.5), ran(0.2)
        "The bird"   → flew(0.6), sang(0.3)
→ Giữ 3 tốt nhất: "The bird flew", "The dog barked", "The cat sat"
```

Beam search cho kết quả **ổn định hơn** nhưng **ít đa dạng** hơn sampling. Trong repo, `evaluation.py` dùng `num_beams` cho predict.

### 13.17 Chat Template & Special Tokens

**Chat Template** là cách format input cho model chat. Mỗi model family có template riêng.

**ChatGLM2:**
```
[gMASK]sop [Round 1]

问：Please detect malware traffic.

答：Malware Traffic Detection
```

**BitNet (LLaMA 3 style):**
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Please detect malware traffic.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Malware Traffic Detection<|eot_id|>
```

**Special Tokens:**

| Token | Ý nghĩa | Mục đích |
|-------|---------|---------|
| `[EOS]` / `<\|eot_id\|>` | End of Sequence | Báo model ngừng sinh text |
| `[PAD]` | Padding | Đệm cho batch cùng độ dài |
| `[gMASK]` | Generation Mask | ChatGLM2: bắt đầu generation |
| `sop` | Start of Piece | ChatGLM2: bắt đầu token |

Trong training, labels cho phần **input** được set thành `-100` → loss function bỏ qua, chỉ tính loss cho phần **output** (đáp án):

```
Input tokens:  [user_msg_tokens...]  [assistant_header]
Labels:        [-100, -100, ...]     [-100, ...]

Output tokens: [Malware Traffic Detection]  [EOS]
Labels:        [token_ids thật...]          [EOS_id]
               ↑ chỉ tính loss ở đây
```

### 13.18 Quantization & Ternary Weights (BitNet)

**Quantization** là kỹ thuật **giảm kích thước model** bằng cách giảm độ chính xác số.

```
FP32 weight:  0.0234375            → 32 bits per param
FP16 weight:  0.02344              → 16 bits per param
INT8 weight:  6 (= 0.023 × 256)   → 8 bits per param
INT4 weight:  1 (= 0.023 × 64)    → 4 bits per param
```

**BitNet b1.58 — Ternary Quantization:**

Cực đoan nhất: mỗi weight chỉ có **3 giá trị**: {-1, 0, +1}

```
Weight matrix bình thường:
  [ 0.23  -0.41   0.05  -0.67   0.12 ]
  [-0.33   0.89  -0.02   0.45  -0.78 ]

Sau ternary quantization:
  [ 0  -1   0  -1   0 ]        → 1.58 bits per param
  [-1   1   0   1  -1 ]           (log₂(3) ≈ 1.58)
```

**Lợi ích:** Phép nhân ma trận `y = Wx` trở thành **phép cộng/trừ** (không cần nhân thực sự) → nhanh hơn rất nhiều trên phần cứng tối ưu.

**Lưu ý quan trọng** (ghi trong README): HuggingFace `transformers` **KHÔNG** tận dụng tốc độ ternary. Cần dùng `bitnet.cpp` runtime để đạt hiệu quả thực sự. Training dùng BF16 master weights (`microsoft/bitnet-b1.58-2B-4T-bf16`).

### 13.19 Distributed Training (DDP, torchrun)

Khi model lớn hoặc data nhiều, dùng **nhiều GPU** để train song song.

**DDP (Distributed Data Parallel):**

```
GPU 0: Model copy → Batch 1 → Gradient 1 ─┐
GPU 1: Model copy → Batch 2 → Gradient 2 ─┼─→ Average gradients → Update ALL
GPU 2: Model copy → Batch 3 → Gradient 3 ─┘

→ Mỗi GPU xử lý 1 phần data, gradient được tổng hợp → update đồng bộ
→ Effective batch size = batch_per_gpu × num_gpus
```

Trong repo:
```bash
NUM_GPUS=1
torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py ...
```

- `--standalone`: chạy trên 1 máy (không phải cluster)
- `--nnodes=1`: 1 node
- `--nproc-per-node=1`: 1 GPU per node (tăng nếu có nhiều GPU)

### 13.20 Evaluation: ROUGE & BLEU (cho Generation)

Dùng trong `dual-stage-tuning/main.py` để đánh giá **traffic generation tasks**.

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation):**

Đo mức trùng lặp n-gram giữa text sinh ra và đáp án:

```
Reference: "the cat sat on the mat"
Predicted: "the cat is on the mat"

ROUGE-1 (unigram overlap):
  Reference words: {the, cat, sat, on, the, mat} = {the, cat, sat, on, mat}
  Predicted words: {the, cat, is, on, the, mat} = {the, cat, is, on, mat}
  Overlap: {the, cat, on, mat} = 4/5 = 0.8

ROUGE-2 (bigram overlap):
  Reference bigrams: {the-cat, cat-sat, sat-on, on-the, the-mat}
  Predicted bigrams: {the-cat, cat-is, is-on, on-the, the-mat}
  Overlap: {the-cat, on-the, the-mat} = 3/5 = 0.6

ROUGE-L (longest common subsequence):
  LCS("the cat sat on the mat", "the cat is on the mat") = "the cat on the mat" (5)
  ROUGE-L = 5/6 = 0.83
```

**BLEU (Bilingual Evaluation Understudy):**

Tương tự ROUGE nhưng thiên về **precision** (chính xác) thay vì recall:

```
BLEU-4: Precision dựa trên 4-gram overlap
Smoothing: tránh BLEU=0 khi không có 4-gram trùng
```

Trong code:
```python
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

rouge = Rouge()
scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))
bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
```

### 13.21 Confusion Matrix & Classification Report

Dùng trong `evaluation.py` và `evaluation_bitnet.py` cho **detection tasks**.

**Confusion Matrix** (Ma trận nhầm lẫn):

Với 3 class {A, B, C} và 10 samples:

```
                    Predicted
                 A    B    C
Actual    A  [  3    1    0  ]   ← 3 đúng A, 1 nhầm thành B
          B  [  0    2    1  ]   ← 2 đúng B, 1 nhầm thành C
          C  [  1    0    2  ]   ← 2 đúng C, 1 nhầm thành A

Đường chéo = dự đoán đúng
Ngoài đường chéo = dự đoán sai
```

**Classification Report** (sklearn):

```
              precision    recall  f1-score   support

     Zeus       0.95      0.92      0.93       120
    Neris       0.88      0.91      0.89        85
   Cridex       0.72      0.68      0.70        45
    ...

 accuracy                           0.87       500
macro avg       0.85      0.84      0.84       500
weighted avg    0.87      0.87      0.87       500
```

- **support**: số sample thực tế cho mỗi class
- **macro avg**: trung bình không trọng số (mỗi class bình đẳng)
- **weighted avg**: trung bình có trọng số theo support (class lớn ảnh hưởng nhiều hơn)

---

### Tổng kết: Map thuật toán → File trong repo

| Thuật toán / Khái niệm | Dùng ở đâu trong repo |
|------------------------|----------------------|
| Transformer, Self-Attention, MHA | Bên trong model ChatGLM2 / BitNet / Llama (pre-trained) |
| GQA (Grouped Query Attention) | BitNet b1.58-2B-4T architecture |
| P-Tuning v2 (Prefix Tuning) | `dual-stage-tuning/main.py`, `inference.py`, `evaluation.py` |
| LoRA (Low-Rank Adaptation) | `Adapt2BitNet/FT/finetune.py`, `Adapt2GLM4/FT/finetune.py` |
| BPE Tokenization | `tokenization/traffic_tokenizer.py` |
| Cross-Entropy Loss | Implicit trong Trainer (HuggingFace) |
| Backpropagation | Implicit trong `loss.backward()` (PyTorch) |
| Gradient Accumulation | Training scripts: `--gradient_accumulation_steps 16` |
| Gradient Checkpointing | `model.gradient_checkpointing_enable()` trong `main.py`, `finetune.py` |
| Mixed Precision (FP16/BF16) | `model.half()` (ChatGLM2), `bf16: true` (BitNet lora.yaml) |
| Temperature, Top-p | `trafficllm_server.py` (Streamlit sliders), `evaluation.py`, `evaluation_bitnet.py` |
| Beam Search | `main.py`: `--generation_num_beams` |
| Chat Template | `inference_bitnet.py`: `tokenizer.apply_chat_template()` |
| Ternary Quantization | BitNet model weights (pre-trained, không trong repo code) |
| DDP / torchrun | `trafficllm_stage1.sh`, `trafficllm_stage2.sh` |
| ROUGE, BLEU-4 | `dual-stage-tuning/main.py`: `compute_metrics()` |
| Confusion Matrix | `evaluation.py`, `evaluation_bitnet.py`: `td_evaluation()` |
| Accuracy, Precision, Recall, F1 | `evaluation.py`, `evaluation_bitnet.py`: `sklearn.metrics` |
| Softmax | Implicit trong model output layer, attention scores |
| Warmup + LR Scheduler | `lora.yaml`: `warmup_steps: 200`, Trainer auto-handles |
| SwiGLU FFN | BitNet/Llama architecture: `gate_proj`, `up_proj`, `down_proj` |
| RoPE (Rotary Position) | BitNet/Llama architecture (internal) |

---

> **Ghi chú cuối**: Repo này đang phát triển tích cực. Các backbone model được hỗ trợ bao gồm ChatGLM2, GLM4, BitNet, Llama2, và DeepSeek-R1. Mỗi backbone có cách fine-tune và inference riêng nhưng chia sẻ cùng pipeline dữ liệu và kiến trúc dual-stage.
