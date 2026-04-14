"""
BitNet b1.58-2B-4T evaluation for TrafficLLM traffic tasks.

Mirrors evaluation.py metrics (accuracy, precision, recall, F1, confusion matrix)
using BitNet's model.generate() instead of ChatGLM2's .chat().

Usage:
  python evaluation_bitnet.py \
      --model_name microsoft/bitnet-b1.58-2B-4T-bf16 \
      --test_file ../datasets/ustc-tfc-2016/ustc-tfc-2016_detection_packet_test.json \
      --label_file ../datasets/ustc-tfc-2016/ustc-tfc-2016_detection_packet_label.json \
      --adapter_path ../models/bitnet/adapters/ustc-tfc-2016-detection-packet \
      --traffic_task detection
"""

import json
import os
import sys

import fire
import torch
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_set_to_prompt(test_set):
    test_prompts = []
    target_responses = []

    for test_data in test_set:
        obj = json.loads(test_data)
        test_prompts.append(obj["instruction"])
        target_responses.append(obj["output"])

    return test_prompts, target_responses


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


def td_evaluation(predict_responses, target_responses, label_file):
    """Traffic Detection evaluation with classification metrics."""
    with open(label_file, "r", encoding="utf-8") as fin:
        label_dict = json.load(fin)

    preds = []
    labels = []
    unknown_label = len(label_dict)

    for predict_response, target_response in zip(predict_responses, target_responses):
        if predict_response.endswith("\u3002"):
            predict_response = predict_response[:-1]
        if target_response.endswith("\u3002"):
            target_response = target_response[:-1]

        if " " not in predict_response:
            pred_key = predict_response
        else:
            pred_key = predict_response.split(" ")[-1]
            if pred_key.endswith("\u3002"):
                pred_key = pred_key[:-1]

        if " " not in target_response:
            label_key = target_response
        else:
            label_key = target_response.split(" ")[-1]
            if label_key.endswith("\u3002"):
                label_key = label_key[:-1]

        if pred_key not in label_dict:
            preds.append(unknown_label)
            print(f"Unknown prediction: '{pred_key}'")
        else:
            preds.append(label_dict[pred_key])

        labels.append(label_dict[label_key])

    print(f"\n{'='*60}")
    print(f"Traffic Detection Evaluation Results")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy_score(labels, preds):.4f}")
    print(f"Precision: {precision_score(labels, preds, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(labels, preds, average='weighted', zero_division=0):.4f}")
    print(f"F1:        {f1_score(labels, preds, average='weighted', zero_division=0):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(labels, preds)}")
    print(f"\nClassification Report:\n{classification_report(labels, preds, zero_division=0)}")


def tg_evaluation(predict_responses, target_responses, test_prompts):
    """Traffic Generation evaluation -- saves generated outputs grouped by label."""
    write_path = "generation_bitnet.json"
    dataset = {}
    for predict_response, target_response, test_prompt in zip(
        predict_responses, target_responses, test_prompts
    ):
        label = test_prompt.split(" ")[-2]
        if label not in dataset:
            dataset[label] = []
        dataset[label].append(predict_response)

    with open(write_path, "w", encoding="utf-8") as fin:
        json.dump(dataset, fin, indent=4, separators=(",", ": "))

    print(f"Generation results saved to {write_path}")
    print(f"Labels found: {list(dataset.keys())}")
    print(f"Total samples: {sum(len(v) for v in dataset.values())}")


def main(
    model_name: str = "microsoft/bitnet-b1.58-2B-4T-bf16",
    test_file: str = None,
    label_file: str = None,
    traffic_task: str = None,
    adapter_path: str = None,
    max_samples: int = 1000,
    device: str = "cuda",
):
    if test_file is None or not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        sys.exit(1)

    with open(test_file, "r", encoding="utf-8") as fin:
        test_set = fin.readlines()

    print(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    if adapter_path is not None:
        print(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    test_prompts, target_responses = test_set_to_prompt(test_set)
    test_prompts = test_prompts[:max_samples]
    target_responses = target_responses[:max_samples]

    print(f"Running inference on {len(test_prompts)} samples...")
    predict_responses = []

    for test_prompt in tqdm(test_prompts, desc="Evaluating"):
        if traffic_task == "detection":
            response = generate_response(
                model, tokenizer, test_prompt,
                max_new_tokens=64, temperature=0.1, top_p=0.85,
            )
        elif traffic_task == "generation":
            response = generate_response(
                model, tokenizer, test_prompt,
                max_new_tokens=512, temperature=0.7, top_p=0.9,
            )
        else:
            response = generate_response(model, tokenizer, test_prompt)
        predict_responses.append(response)

    if traffic_task == "detection":
        td_evaluation(predict_responses, target_responses, label_file)
    elif traffic_task == "generation":
        tg_evaluation(predict_responses, target_responses, test_prompts)
    else:
        print("Unknown traffic_task. Use 'detection' or 'generation'.")


if __name__ == "__main__":
    fire.Fire(main)
