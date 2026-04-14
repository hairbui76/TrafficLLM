"""
BitNet b1.58-2B-4T fine-tuning for TrafficLLM.

Uses the BF16 variant (microsoft/bitnet-b1.58-2B-4T-bf16) with PEFT LoRA
for dual-stage traffic analysis training. Adapted from Adapt2GLM4/FT/finetune.py.
"""

import logging
import sys
import os
import functools
import dataclasses as dc
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Optional, Union

import numpy as np
import ruamel.yaml as yaml
import torch
import typer
from datasets import Dataset, DatasetDict, NamedSplit, Split, load_dataset
from peft import PeftConfig, get_peft_config, get_peft_model
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
    GenerationConfig,
    PreTrainedTokenizer,
    Seq2SeqTrainingArguments,
)
from transformers import DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer as _Seq2SeqTrainer
from sklearn.metrics import accuracy_score, f1_score

app = typer.Typer(pretty_exceptions_show_locals=False)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "microsoft/bitnet-b1.58-2B-4T-bf16"


class DataCollatorForSeq2Seq(_DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        output_ids = (
            [feature["output_ids"] for feature in features]
            if "output_ids" in features[0].keys()
            else None
        )
        if output_ids is not None:
            max_output_length = max(len(out) for out in output_ids)
            if self.pad_to_multiple_of is not None:
                max_output_length = (
                    (max_output_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (
                    max_output_length - len(feature["output_ids"])
                )
                if isinstance(feature["output_ids"], list):
                    feature["output_ids"] = feature["output_ids"] + remainder
                else:
                    feature["output_ids"] = np.concatenate(
                        [feature["output_ids"], remainder]
                    ).astype(np.int64)
        return super().__call__(features, return_tensors)


class Seq2SeqTrainer(_Seq2SeqTrainer):
    def training_step(self, model: nn.Module, inputs: dict[str, Any]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()
        self.accelerator.backward(loss)
        detached_loss = loss.detach() / self.args.gradient_accumulation_steps
        del inputs
        torch.cuda.empty_cache()
        return detached_loss

    def log(self, logs: dict) -> None:
        logger.info(f"Logs: {logs}")
        super().log(logs)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys=None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            if self.args.predict_with_generate:
                output_ids = inputs.pop("output_ids")
            input_ids = inputs["input_ids"]

            loss, generated_tokens, labels = super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs
            )

            generated_tokens = generated_tokens[:, input_ids.size()[1] :]
            labels = output_ids

            del inputs, input_ids, output_ids
            torch.cuda.empty_cache()

        return loss, generated_tokens, labels


@dc.dataclass
class DataConfig:
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None
    num_proc: Optional[int] = None

    @property
    def data_format(self) -> str:
        return Path(self.train_file).suffix

    @property
    def data_files(self) -> dict[NamedSplit, str]:
        return {
            split: data_file
            for split, data_file in zip(
                [Split.TRAIN, Split.VALIDATION, Split.TEST],
                [self.train_file, self.val_file, self.test_file],
            )
            if data_file is not None
        }


@dc.dataclass
class FinetuningConfig:
    data_config: DataConfig
    max_input_length: int
    max_output_length: int
    combine: bool
    training_args: Seq2SeqTrainingArguments = dc.field(
        default_factory=lambda: Seq2SeqTrainingArguments(output_dir="./output")
    )
    peft_config: Optional[PeftConfig] = None

    def __post_init__(self):
        if not self.training_args.do_eval or self.data_config.val_file is None:
            self.training_args.do_eval = False
            self.training_args.evaluation_strategy = "no"
            self.data_config.val_file = None
        else:
            self.training_args.per_device_eval_batch_size = (
                self.training_args.per_device_eval_batch_size
                or self.training_args.per_device_train_batch_size
            )

    @classmethod
    def from_dict(cls, **kwargs) -> "FinetuningConfig":
        training_args = kwargs.get("training_args", None)
        if training_args is not None and not isinstance(
            training_args, Seq2SeqTrainingArguments
        ):
            gen_config = training_args.get("generation_config")
            if not isinstance(gen_config, GenerationConfig):
                training_args["generation_config"] = GenerationConfig(**gen_config)
            kwargs["training_args"] = Seq2SeqTrainingArguments(**training_args)

        data_config = kwargs.get("data_config")
        if not isinstance(data_config, DataConfig):
            kwargs["data_config"] = DataConfig(**data_config)

        peft_config = kwargs.get("peft_config", None)
        if peft_config is not None and not isinstance(peft_config, PeftConfig):
            kwargs["peft_config"] = get_peft_config(config_dict=peft_config)
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "FinetuningConfig":
        path = Path(path)
        parser = yaml.YAML(typ="safe", pure=True)
        parser.indent(mapping=2, offset=2, sequence=4)
        parser.default_flow_style = False
        kwargs = parser.load(path)
        return cls.from_dict(**kwargs)


def _load_datasets(
    data_dir: str,
    data_format: str,
    data_files: dict[NamedSplit, str],
    num_proc: Optional[int],
) -> DatasetDict:
    if data_format in (".jsonl", ".json"):
        dataset_dct = load_dataset(
            data_dir,
            data_files=data_files,
            split=None,
            num_proc=num_proc,
        )
    else:
        raise NotImplementedError(f"Cannot load dataset in the '{data_format}' format.")
    return dataset_dct


class DataManager:
    def __init__(self, data_dir: str, data_config: DataConfig):
        self._num_proc = data_config.num_proc
        self._dataset_dct = _load_datasets(
            data_dir,
            data_config.data_format,
            data_config.data_files,
            self._num_proc,
        )

    def _get_dataset(self, split: NamedSplit) -> Optional[Dataset]:
        return self._dataset_dct.get(split, None)

    def get_dataset(
        self,
        split: NamedSplit,
        process_fn: Callable[[dict[str, Any]], dict[str, Any]],
        batched: bool = True,
        remove_orig_columns: bool = True,
    ) -> Optional[Dataset]:
        orig_dataset = self._get_dataset(split)
        if orig_dataset is None:
            return None

        if remove_orig_columns:
            remove_columns = orig_dataset.column_names
        else:
            remove_columns = None
        return orig_dataset.map(
            process_fn,
            batched=batched,
            remove_columns=remove_columns,
            num_proc=self._num_proc,
        )


def build_chat_input(
    tokenizer: PreTrainedTokenizer,
    user_content: str,
    assistant_content: str,
    max_input_length: int,
    max_output_length: int,
) -> tuple[list[int], list[int]]:
    """Build input_ids and labels from a single user/assistant turn using the
    tokenizer's chat template (LLaMA 3 style for BitNet)."""
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )

    user_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=True,
        add_generation_prompt=True,
    )
    user_len = len(user_only)

    max_length = max_input_length + max_output_length
    full_ids = full_ids[:max_length]

    labels = [-100] * min(user_len, len(full_ids))
    if len(full_ids) > user_len:
        labels += full_ids[user_len:]

    return full_ids, labels


def process_batch(
    batch: Mapping[str, Sequence],
    tokenizer: PreTrainedTokenizer,
    max_input_length: int,
    max_output_length: int,
    combine: bool,
) -> dict[str, list]:
    """Process a batch of training examples. Supports both 'messages' format
    (chat-style) and 'instruction'/'output' format (TrafficLLM-style)."""
    batched_input_ids = []
    batched_labels = []

    if "messages" in batch:
        batched_conv = batch["messages"]
        for conv in batched_conv:
            full_ids = tokenizer.apply_chat_template(
                conv, tokenize=True, add_generation_prompt=False
            )

            labels = list(full_ids)
            assistant_token = tokenizer.encode("assistant", add_special_tokens=False)
            in_assistant = False
            for i, tok in enumerate(full_ids):
                if not in_assistant:
                    labels[i] = -100
                if tok == tokenizer.eos_token_id:
                    in_assistant = False

            max_length = max_input_length + max_output_length
            batched_input_ids.append(full_ids[:max_length])
            batched_labels.append(labels[:max_length])
    else:
        instructions = batch.get("instruction", batch.get("prompt", []))
        outputs = batch.get("output", batch.get("response", []))
        for instruction, output in zip(instructions, outputs):
            if not instruction or not output:
                continue
            input_ids, labels = build_chat_input(
                tokenizer, instruction, output,
                max_input_length, max_output_length,
            )
            batched_input_ids.append(input_ids)
            batched_labels.append(labels)

    return {"input_ids": batched_input_ids, "labels": batched_labels}


def process_batch_eval(
    batch: Mapping[str, Sequence],
    tokenizer: PreTrainedTokenizer,
    max_input_length: int,
    max_output_length: int,
    combine: bool,
) -> dict[str, list]:
    """Process a batch of evaluation examples."""
    batched_input_ids = []
    batched_output_ids = []

    if "messages" in batch:
        batched_conv = batch["messages"]
        for conv in batched_conv:
            user_messages = [m for m in conv if m["role"] != "assistant"]
            assistant_messages = [m for m in conv if m["role"] == "assistant"]

            input_ids = tokenizer.apply_chat_template(
                user_messages, tokenize=True, add_generation_prompt=True
            )
            if assistant_messages:
                output_ids = tokenizer.encode(
                    assistant_messages[-1]["content"], add_special_tokens=False
                )
                output_ids.append(tokenizer.eos_token_id)
            else:
                output_ids = [tokenizer.eos_token_id]

            batched_input_ids.append(input_ids[:max_input_length])
            batched_output_ids.append(output_ids[:max_output_length])
    else:
        instructions = batch.get("instruction", batch.get("prompt", []))
        outputs = batch.get("output", batch.get("response", []))
        for instruction, output in zip(instructions, outputs):
            if not instruction or not output:
                continue
            messages = [{"role": "user", "content": instruction}]
            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            output_ids = tokenizer.encode(output, add_special_tokens=False)
            output_ids.append(tokenizer.eos_token_id)

            batched_input_ids.append(input_ids[:max_input_length])
            batched_output_ids.append(output_ids[:max_output_length])

    return {"input_ids": batched_input_ids, "output_ids": batched_output_ids}


def load_tokenizer_and_model(
    model_dir: str,
    peft_config: Optional[PeftConfig] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(
        trust_remote_code=True,
        use_cache=False,
        torch_dtype=torch.bfloat16,
    )

    if peft_config is not None:
        model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)

    return tokenizer, model


def compute_metrics(eval_preds: EvalPrediction, tokenizer: PreTrainedTokenizer):
    batched_pred_ids, batched_label_ids = eval_preds
    metrics_dct = {"exact_match": [], "f1_token": []}
    for pred_ids, label_ids in zip(batched_pred_ids, batched_label_ids):
        pred_txt = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
        label_txt = tokenizer.decode(label_ids, skip_special_tokens=True).strip()
        metrics_dct["exact_match"].append(1.0 if pred_txt == label_txt else 0.0)

        pred_tokens = set(pred_txt.lower().split())
        label_tokens = set(label_txt.lower().split())
        if len(pred_tokens) == 0 and len(label_tokens) == 0:
            metrics_dct["f1_token"].append(1.0)
        elif len(pred_tokens) == 0 or len(label_tokens) == 0:
            metrics_dct["f1_token"].append(0.0)
        else:
            common = pred_tokens & label_tokens
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(label_tokens)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            metrics_dct["f1_token"].append(f1)

    return {k: float(np.mean(v)) for k, v in metrics_dct.items()}


@app.command()
def main(
    data_dir: Annotated[str, typer.Argument(help="Path to directory containing train/val/test JSONL files")],
    model_dir: Annotated[
        str,
        typer.Argument(
            help="HuggingFace model ID or local path. Default: microsoft/bitnet-b1.58-2B-4T-bf16"
        ),
    ] = DEFAULT_MODEL_ID,
    config_file: Annotated[str, typer.Argument(help="Path to LoRA YAML config file")] = "configs/lora.yaml",
    auto_resume_from_checkpoint: str = typer.Argument(
        default="",
        help="'yes' to auto-resume from latest checkpoint, a number to resume from specific checkpoint, empty to start fresh",
    ),
):
    ft_config = FinetuningConfig.from_file(config_file)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    import transformers
    if ft_config.training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = ft_config.training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {ft_config.training_args.local_rank}, "
        f"device: {ft_config.training_args.device}, "
        f"n_gpu: {ft_config.training_args.n_gpu}, "
        f"distributed training: {bool(ft_config.training_args.local_rank != -1)}, "
        f"bf16 training: {ft_config.training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {ft_config.training_args}")

    tokenizer, model = load_tokenizer_and_model(model_dir, peft_config=ft_config.peft_config)
    data_manager = DataManager(data_dir, ft_config.data_config)

    train_dataset = data_manager.get_dataset(
        Split.TRAIN,
        functools.partial(
            process_batch,
            tokenizer=tokenizer,
            combine=ft_config.combine,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    logger.info(f"Train dataset: {train_dataset}")

    val_dataset = data_manager.get_dataset(
        Split.VALIDATION,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            combine=ft_config.combine,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )

    test_dataset = data_manager.get_dataset(
        Split.TEST,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            combine=ft_config.combine,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    trainer = Seq2SeqTrainer(
        model=model,
        args=ft_config.training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding="longest",
            return_tensors="pt",
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer),
    )

    if not auto_resume_from_checkpoint or auto_resume_from_checkpoint.upper() == "":
        logger.info("Starting training from scratch")
        trainer.train()
        trainer.save_state()
    else:
        output_dir = ft_config.training_args.output_dir
        dirlist = os.listdir(output_dir) if os.path.exists(output_dir) else []
        checkpoint_sn = 0
        for checkpoint_str in dirlist:
            if "eckpoint" in checkpoint_str and "tmp" not in checkpoint_str:
                checkpoint = int(checkpoint_str.replace("checkpoint-", ""))
                if checkpoint > checkpoint_sn:
                    checkpoint_sn = checkpoint

        if auto_resume_from_checkpoint.upper() == "YES":
            if checkpoint_sn > 0:
                checkpoint_directory = os.path.join(output_dir, f"checkpoint-{checkpoint_sn}")
                logger.info(f"Resuming from {checkpoint_directory}")
                trainer.train(resume_from_checkpoint=checkpoint_directory)
            else:
                logger.info("No checkpoint found, starting from scratch")
                trainer.train()
        elif auto_resume_from_checkpoint.isdigit():
            checkpoint_sn = int(auto_resume_from_checkpoint)
            checkpoint_directory = os.path.join(output_dir, f"checkpoint-{checkpoint_sn}")
            if os.path.exists(checkpoint_directory):
                logger.info(f"Resuming from {checkpoint_directory}")
                trainer.train(resume_from_checkpoint=checkpoint_directory)
            else:
                logger.error(
                    f"Checkpoint {checkpoint_directory} not found. "
                    f"Available checkpoints in {output_dir}: {dirlist}"
                )
        trainer.save_state()


if __name__ == "__main__":
    app()
