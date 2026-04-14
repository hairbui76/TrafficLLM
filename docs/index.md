# TrafficLLM Documentation

**TrafficLLM: Enhancing Large Language Models for Network Traffic Analysis with Robust Traffic Representation**

- Paper: [arXiv:2504.04222](https://arxiv.org/abs/2504.04222)
- GitHub: [ZGC-LLM-Safety/TrafficLLM](https://github.com/ZGC-LLM-Safety/TrafficLLM)
- Datasets: [Google Drive](https://drive.google.com/drive/folders/1RZAOPcNKq73-quA8KG_lkAo_EqlwhlQb)
- Models: [Google Drive](https://drive.google.com/drive/folders/1YjEhdordqZRpnw_oKczwUztcT52T0oQ0)

TrafficLLM is a universal LLM adaptation framework that learns robust traffic representations for open-sourced LLMs in real-world network scenarios. It bridges the modality gap between natural language and heterogeneous traffic data through traffic-domain tokenization, a dual-stage tuning pipeline for instruction understanding and task-specific pattern learning, and an extensible adaptation mechanism (EA-PEFT) for low-overhead updates to new traffic environments. The framework supports multiple LLM backends (ChatGLM2, Llama, GLM-4, BitNet) across 14 traffic analysis tasks spanning detection and generation.

## Table of Contents

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | System architecture, directory structure, dual-stage pipeline, config files, task registry |
| [Data Pipeline](data-pipeline.md) | PCAP preprocessing, data formats, dataset catalog, tokenization |
| [Model Backends](model-backends.md) | Side-by-side comparison of ChatGLM2, Llama, GLM-4, and BitNet backends |
| [Training Guide](training-guide.md) | Step-by-step training instructions for every backend |
| [Inference and Deployment](inference-and-deployment.md) | CLI inference, Streamlit demo, agent systems, evaluation, traffic generation |
| [BitNet Integration](bitnet-integration.md) | Detailed documentation of the BitNet b1.58-2B-4T ternary LLM integration |
| [EA-PEFT](ea-peft.md) | Extensible Adaptation with Parameter-Efficient Fine-Tuning mechanism |
| [Developer Guide](developer-guide.md) | How to add new model backbones, new traffic tasks, and project conventions |

## Quick Start

### Environment Setup

```bash
conda create -n trafficllm python=3.9
conda activate trafficllm
git clone https://github.com/ZGC-LLM-Safety/TrafficLLM.git
cd TrafficLLM
pip install -r requirements.txt
pip install rouge_chinese nltk jieba datasets
```

### Minimal Training (ChatGLM2)

```bash
cd dual-stage-tuning
bash trafficllm_stage1.sh   # Stage 1: instruction tuning
bash trafficllm_stage2.sh   # Stage 2: task-specific tuning
```

### Minimal Training (BitNet)

```bash
cd Adapt2BitNet/FT/scripts
bash trafficllm_stage1.sh   # Stage 1: instruction tuning with LoRA
bash trafficllm_stage2.sh ustc-tfc-2016-detection-packet \
    ../../datasets/ustc-tfc-2016/ustc-tfc-2016_detection_packet_train.json
```

### Minimal Inference

```bash
# ChatGLM2
python inference.py --config=config.json \
    --prompt="Please help me detect malware traffic.<packet>frame.encap_type: 1, ..."

# BitNet
python Adapt2BitNet/inference_bitnet.py --config=Adapt2BitNet/config.json \
    --prompt="Please help me detect malware traffic.<packet>frame.encap_type: 1, ..."
```

### Web Demo

```bash
streamlit run trafficllm_server.py
```
