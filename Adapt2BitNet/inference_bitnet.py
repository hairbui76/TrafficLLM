"""
BitNet b1.58-2B-4T dual-stage inference for TrafficLLM.

Replaces ChatGLM2's .chat() + prefix_encoder pattern with
AutoModelForCausalLM + model.generate() + LoRA adapter hot-swap.

Usage:
  python inference_bitnet.py --config config.json \
      --prompt "Please help me detect malware traffic.<packet>frame.encap_type: 1, ..."
"""

import json
import os
from typing import Optional

import fire
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_base_model(model_path: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    return tokenizer, model


def load_adapter(model, adapter_path: str) -> PeftModel:
    """Load a LoRA adapter on top of the base model."""
    return PeftModel.from_pretrained(model, adapter_path)


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.1,
    top_p: float = 0.85,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    if isinstance(input_ids, list):
        input_ids = torch.tensor([input_ids])
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response


def prompt_processing(prompt: str):
    """Split user input into instruction text and traffic data."""
    if "<packet>" in prompt:
        instruction_text = prompt.split("<packet>")[0].strip()
        traffic_data = "<packet>" + "<packet>".join(prompt.split("<packet>")[1:])
    else:
        instruction_text = prompt
        traffic_data = ""
    return instruction_text, traffic_data


def preprompt(task: str, traffic_data: str) -> str:
    """Task-specific preprompts for downstream traffic pattern learning."""
    preprompt_set = {
        "MTD": (
            "Given the following traffic data <packet> that contains protocol fields, "
            "traffic features, and payloads. Please conduct the ENCRYPTED MALWARE "
            "DETECTION TASK to determine which application category the encrypted "
            "benign or malicious traffic belongs to. The categories include "
            "'BitTorrent, FTP, Facetime, Gmail, MySQL, Outlook, SMB, Skype, Weibo, "
            "WorldOfWarcraft, Cridex, Geodo, Htbot, Miuref, Neris, Nsis-ay, Shifu, "
            "Tinba, Virut, Zeus'.\n"
        ),
        "BND": (
            "Given the following traffic data <packet> that contains protocol fields, "
            "traffic features, and payloads. Please conduct the BOTNET DETECTION TASK "
            "to determine which type of network the traffic belongs to. The categories "
            "include 'IRC, Neris, RBot, Virut, normal'.\n"
        ),
        "WAD": (
            "Classify the given HTTP request into benign and malicious categories. Each "
            "HTTP request will consist of three parts: method, URL, and body, presented "
            "in JSON format. If a web attack is detected in an HTTP request, please "
            "output an 'exception'. Only output 'malicious' or 'benign', no additional "
            "output is required. The given HTTP request is as follows:\n"
        ),
        "AAD": (
            "Classify the given HTTP request into normal and abnormal categories. Each "
            "HTTP request will consist of three parts: method, URL, and body, presented "
            "in JSON format. If a web attack is detected in an HTTP request, please "
            "output an 'exception'. Only output 'abnormal' or 'normal', no additional "
            "output is required. The given HTTP request is as follows:\n"
        ),
        "EVD": (
            "Given the following traffic data <packet> that contains protocol fields, "
            "traffic features, and payloads. Please conduct the encrypted VPN detection "
            "task to determine which behavior or application category the VPN encrypted "
            "traffic belongs to. The categories include 'aim, bittorrent, email, "
            "facebook, ftps, hangout, icq, netflix, sftp, skype, spotify, vimeo, "
            "voipbuster, youtube'.\n"
        ),
        "TBD": (
            "Given the following traffic data <packet> that contains protocol fields, "
            "traffic features, and payloads. Please conduct the TOR BEHAVIOR DETECTION "
            "TASK to determine which behavior or application category the traffic "
            "belongs to under the Tor network. The categories include 'audio, browsing, "
            "chat, file, mail, p2p, video, voip'.\n"
        ),
    }
    if task == "AAD":
        prompt = preprompt_set[task] + traffic_data.split("<packet>:")[1]
    else:
        prompt = preprompt_set[task] + traffic_data
    return prompt


def main(
    config: str = "config.json",
    prompt: Optional[str] = None,
    device: str = "cuda",
):
    instruction_text, traffic_data = prompt_processing(prompt)

    with open(config, "r", encoding="utf-8") as fin:
        cfg = json.load(fin)

    print(f"Loading base model: {cfg['model_path']}")
    tokenizer, base_model = load_base_model(cfg["model_path"], device=device)

    # Stage 1: Task understanding via NLP adapter
    nlp_adapter_path = os.path.join(cfg["adapter_path"], cfg["adapter_set"]["NLP"])
    print(f"Loading NLP adapter: {nlp_adapter_path}")
    model_nlp = load_adapter(base_model, nlp_adapter_path)
    model_nlp.eval()

    response = generate_response(model_nlp, tokenizer, instruction_text)
    print(f"Stage 1 - Detected task: {response}")

    # Unload NLP adapter to free memory before loading task adapter
    model_nlp = model_nlp.unload()

    # Stage 2: Task-specific traffic analysis
    if response not in cfg["tasks"]:
        print(f"Warning: Unrecognized task '{response}'. Available tasks: {list(cfg['tasks'].keys())}")
        return

    task = cfg["tasks"][response]
    task_adapter_path = os.path.join(cfg["adapter_path"], cfg["adapter_set"][task])
    print(f"Loading task adapter ({task}): {task_adapter_path}")
    model_task = load_adapter(base_model, task_adapter_path)
    model_task.eval()

    traffic_prompt = preprompt(task, traffic_data)
    result = generate_response(model_task, tokenizer, traffic_prompt)
    print(f"Stage 2 - Result: {result}")

    return result


if __name__ == "__main__":
    fire.Fire(main)
