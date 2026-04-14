"""
Convert TrafficLLM's instruction/output JSONL format to chat messages JSONL
format compatible with BitNet's LLaMA 3 chat template.

Input format (one JSON object per line):
  {"instruction": "...", "output": "..."}

Output format (one JSON object per line):
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Usage:
  python convert_to_chat.py --input_file ../datasets/instructions/instructions.json \
                            --output_file ../datasets/instructions/instructions_chat.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def convert_line(obj: dict) -> dict:
    instruction = obj.get("instruction", obj.get("prompt", ""))
    output = obj.get("output", obj.get("response", ""))
    history = obj.get("history", [])

    messages = []

    for turn in history:
        if isinstance(turn, list) and len(turn) == 2:
            messages.append({"role": "user", "content": turn[0]})
            messages.append({"role": "assistant", "content": turn[1]})
        elif isinstance(turn, dict):
            messages.append(turn)

    messages.append({"role": "user", "content": instruction})
    messages.append({"role": "assistant", "content": output})

    return {"messages": messages}


def convert_file(input_file: str, output_file: str) -> int:
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(input_path, "r", encoding="utf-8") as fin:
        content = fin.read().strip()

        if content.startswith("["):
            data = json.loads(content)
            with open(output_path, "w", encoding="utf-8") as fout:
                for obj in data:
                    converted = convert_line(obj)
                    fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    count += 1
        else:
            with open(output_path, "w", encoding="utf-8") as fout:
                for line in content.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    converted = convert_line(obj)
                    fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Convert instruction/output JSONL to chat messages JSONL"
    )
    parser.add_argument("--input_file", required=True, help="Input JSONL or JSON file")
    parser.add_argument("--output_file", required=True, help="Output JSONL file")
    args = parser.parse_args()

    count = convert_file(args.input_file, args.output_file)
    print(f"Converted {count} examples: {args.input_file} -> {args.output_file}")


if __name__ == "__main__":
    main()
