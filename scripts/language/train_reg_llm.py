import argparse
import os
import copy
import shutil
import tempfile

import torch
from datasets import load_dataset, DatasetDict

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_callback import TrainerCallback
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from peft import LoraConfig, get_peft_model

from lib.regularizer import low_rank_reg_loss

import wandb


def tokenize_function_factory(tokenizer, max_length):
    def fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return fn


class LoRaNuclearRegCallback(TrainerCallback):
    def __init__(self, reg_lambda: float = 1e-1, eps: float = 1e-12):
        self.reg_lambda = reg_lambda
        self.eps = eps
        self.last_reg_loss = None

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs["model"]
        reg_loss = low_rank_reg_loss(model, self.eps, self.reg_lambda)

        if hasattr(reg_loss, "item"):
            reg_loss_value = reg_loss.item() / self.reg_lambda
        else:
            reg_loss_value = reg_loss / self.reg_lambda

        self.last_reg_loss = reg_loss_value

        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and self.last_reg_loss is not None:
            logs["reg_loss"] = self.last_reg_loss
        if wandb.run is not None:
            wandb.log({"reg_loss": self.last_reg_loss}, step=state.global_step)
        return control


class SaveLatestMergedCallback(TrainerCallback):
    def __init__(
        self,
        merged_output_dir: str,
        save_steps: int,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.merged_output_dir = merged_output_dir
        self.save_steps = save_steps
        self.tokenizer = tokenizer
        os.makedirs(self.merged_output_dir, exist_ok=True)

    def _save_merged_copy(self, peft_model):
        clone = copy.deepcopy(peft_model).to("cpu")
        merged = clone.merge_and_unload()

        parent = os.path.dirname(os.path.abspath(self.merged_output_dir)) or "."
        with tempfile.TemporaryDirectory(dir=parent) as tmpdir:
            merged.save_pretrained(tmpdir)
            self.tokenizer.save_pretrained(tmpdir)

            if os.path.exists(self.merged_output_dir):
                shutil.rmtree(self.merged_output_dir)
            os.rename(tmpdir, self.merged_output_dir)

    def on_step_end(self, args, state, control, **kwargs):
        if (
            self.save_steps > 0
            and state.global_step > 0
            and state.global_step % self.save_steps == 0
        ):
            self._save_merged_copy(kwargs["model"])
            print(
                f"[save] wrote latest merged model + tokenizer to {self.merged_output_dir}"
            )
        return control


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3-1b")
    p.add_argument("--dataset_name", type=str, default="wikitext")
    p.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--output_dir", type=str, default="./lora-opt-reg")
    p.add_argument("--merged_output_dir", type=str, default="./lora-merged-final")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--num_train_epochs", type=float, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=1)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--bf16", action="store_true", default=torch.cuda.is_available())
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--reg_lambda", type=float, default=1e-1)
    p.add_argument("--reg_eps", type=float, default=1e-12)
    p.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--wandb_project", type=str, default="lora-opt-reg")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )
    return p.parse_args()


def main():
    args = parse_args()

    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_run_name is not None:
        os.environ["WANDB_RUN_NAME"] = args.wandb_run_name
    os.environ["WANDB_MODE"] = args.wandb_mode

    if args.dataset_config:
        dataset = load_dataset(args.dataset_name, args.dataset_config)
    else:
        dataset = load_dataset(args.dataset_name)

    if "validation" not in dataset:
        split = dataset["train"].train_test_split(test_size=0.005, seed=42)
        dataset = DatasetDict(
            train=split["train"],
            validation=split["test"],
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        legacy=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_datasets = dataset.map(
        tokenize_function_factory(tokenizer, args.max_length),
        batched=True,
        remove_columns=["text"],
    )
    # args.resume=True
    model_name_or_path = (
        args.merged_output_dir
        if args.resume and os.path.isdir(args.merged_output_dir)
        else args.model_name
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
    )

    model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        warmup_steps=0,
        save_strategy="no",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        push_to_hub=args.push_to_hub,
        report_to=["wandb"],
        run_name=args.wandb_run_name,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        callbacks=[
            LoRaNuclearRegCallback(reg_lambda=args.reg_lambda, eps=args.reg_eps),
            SaveLatestMergedCallback(
                merged_output_dir=args.merged_output_dir,
                save_steps=args.save_steps,
                tokenizer=tokenizer,
            ),
        ],
    )

    trainer.train()

    merged = copy.deepcopy(model).merge_and_unload()
    merged.save_pretrained(args.merged_output_dir)
    tokenizer.save_pretrained(args.merged_output_dir)
    print(f"[final] wrote final merged model + tokenizer to {args.merged_output_dir}")


if __name__ == "__main__":
    main()
